# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import asdict
from typing import Optional, Union

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge

    When extra_optimizers is provided (e.g. a separate perception optimizer that owns
    visual-tower parameters), we save each optimizer's native per-rank state_dict
    directly instead of routing optimizer states through torch.distributed.checkpoint.
    In our PyTorch version, get_state_dict/set_state_dict still delegate to
    FSDP.optim_state_dict one optimizer at a time, which breaks under
    use_orig_params=True when the wrapped module is split across multiple optimizers.
    Model states still use the DCP/FSDP APIs so parameter shards remain restorable.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
        extra_optimizers: Optional[list] = None,
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)
        self._extra_optimizers: list = list(extra_optimizers) if extra_optimizers else []

    @property
    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.optimizer] + self._extra_optimizers

    @staticmethod
    def _to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: FSDPCheckpointManager._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [FSDPCheckpointManager._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(FSDPCheckpointManager._to_cpu(item) for item in value)
        return value

    @staticmethod
    def _looks_like_native_optimizer_state_dict(value) -> bool:
        if not isinstance(value, dict):
            return False
        if "state" not in value or "param_groups" not in value:
            return False
        param_groups = value.get("param_groups")
        if not isinstance(param_groups, list) or len(param_groups) == 0:
            return True
        params = param_groups[0].get("params", [])
        return not params or isinstance(params[0], int)

    def _load_native_optimizer_state_dicts(self, payload) -> bool:
        if self._looks_like_native_optimizer_state_dict(payload):
            payload = [payload]
        elif not isinstance(payload, list) or not all(self._looks_like_native_optimizer_state_dict(item) for item in payload):
            return False

        optimizers = self._optimizers
        if len(payload) != len(optimizers):
            raise RuntimeError(
                f"Checkpoint contains {len(payload)} optimizer state dict(s), but current run has {len(optimizers)} optimizer(s)."
            )
        for optimizer, state_dict in zip(optimizers, payload):
            optimizer.load_state_dict(state_dict)
        return True

    def _model_state_dict_options(self) -> StateDictOptions:
        # The perception path enables use_orig_params=True. We have now verified that
        # the checkpoint crash happens inside get_model_state_dict(cpu_offload=True),
        # while the vLLM sync path successfully uses full_state_dict=True without
        # cpu_offload on the same worker. Mirror that path here for perception runs
        # and move tensors to CPU manually before torch.save.
        if self._extra_optimizers:
            return StateDictOptions(full_state_dict=True)
        return StateDictOptions(cpu_offload=True)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = self._model_state_dict_options()
        set_model_state_dict(self.model, model_state_dict, options=state_dict_options)
        if not self._load_native_optimizer_state_dicts(optim_state_dict):
            # Backward compatibility for legacy single-optimizer checkpoints that were
            # saved through torch.distributed.checkpoint before perception loss existed.
            set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict, options=state_dict_options)
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str, save_model_only: bool = False):
        path = self.local_mkdir(path)
        dist.barrier()

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = self._model_state_dict_options()
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            if self._extra_optimizers:
                model_state_dict = self._to_cpu(model_state_dict)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            if self._extra_optimizers:
                model_state_dict = self._to_cpu(model_state_dict)
            optim_state_dict = [self._to_cpu(optimizer.state_dict()) for optimizer in self._optimizers]
            if len(optim_state_dict) == 1:
                optim_state_dict = optim_state_dict[0]
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            assert isinstance(self.model._fsdp_wrapped_module, (PreTrainedModel, PeftModel))
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_path)
            self.model._fsdp_wrapped_module.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        if isinstance(self.model._fsdp_wrapped_module, PeftModel):
            lora_path = os.path.join(path, "lora_adapter")
            peft_config = {}
            if self.rank == 0:
                os.makedirs(lora_path, exist_ok=True)
                peft_config = asdict(self.model._fsdp_wrapped_module.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])

            sharded_lora_weights = get_peft_model_state_dict(
                self.model._fsdp_wrapped_module, state_dict=model_state_dict
            )
            cuda_device = torch.device("cuda")
            lora_weights = {
                name: sharded_weight.to(cuda_device).full_tensor().detach().cpu()
                if isinstance(sharded_weight, DTensor)
                else sharded_weight.detach().cpu()
                for name, sharded_weight in sharded_lora_weights.items()
            }
            torch.cuda.empty_cache()
            if self.rank == 0:
                save_file(lora_weights, os.path.join(lora_path, "adapter_model.safetensors"))
                with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                    json.dump(peft_config, f, ensure_ascii=False, indent=4)

            dist.barrier()
            if self.rank == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_path}")

        dist.barrier()
