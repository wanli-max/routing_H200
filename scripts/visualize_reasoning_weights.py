#!/usr/bin/env python3
"""
Generate responses for the first few validation samples and visualize token weights.

This script is intended for single-GPU inspection with a merged Hugging Face model
under <checkpoint>/actor/huggingface. It reuses the existing rollout + actor routing
logic so the reported weights match the training path as closely as possible.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.config import PPOConfig
from verl.trainer.core_algos import build_effective_token_loss_weights
from verl.utils.answer_localization import localize_answer_token_batch
from verl.utils.dataset import RLHFDataset, collate_fn
from verl.utils.tokenizer import get_processor, get_tokenizer
from verl.workers.fsdp_workers import FSDPWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize response tokens and reasoning weights on 1 GPU.")
    parser.add_argument("--checkpoint", required=True, help="Path to a global_step_* directory or its actor subdir.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store the HTML/JSONL outputs. Defaults beside the checkpoint.",
    )
    parser.add_argument("--max-samples", type=int, default=3, help="How many validation samples to inspect.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional override for generation length.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="vLLM GPU memory utilization for the single-GPU rollout worker.",
    )
    return parser.parse_args()


def resolve_paths(checkpoint_arg: str) -> tuple[Path, Path, Path]:
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if checkpoint_path.name == "actor":
        actor_dir = checkpoint_path
        run_dir = checkpoint_path.parent.parent
    elif checkpoint_path.name.startswith("global_step_"):
        actor_dir = checkpoint_path / "actor"
        run_dir = checkpoint_path.parent
    else:
        raise ValueError("`--checkpoint` must point to a global_step_* directory or its actor subdir.")

    config_path = run_dir / "experiment_config.json"
    merged_model_dir = actor_dir / "huggingface"

    if not actor_dir.exists():
        raise FileNotFoundError(f"Actor checkpoint directory not found: {actor_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_config.json not found: {config_path}")
    if not merged_model_dir.exists():
        raise FileNotFoundError(
            f"Merged huggingface directory not found: {merged_model_dir}. "
            "Run scripts/model_merger.py on the actor checkpoint first."
        )

    model_weight_files = list(merged_model_dir.glob("*.safetensors")) + list(merged_model_dir.glob("pytorch_model*.bin"))
    if not model_weight_files:
        raise FileNotFoundError(
            f"No merged model weights found in {merged_model_dir}. "
            "Run scripts/model_merger.py on the actor checkpoint first."
        )

    return actor_dir, config_path, merged_model_dir


def load_config(config_path: Path, merged_model_dir: Path, args: argparse.Namespace) -> PPOConfig:
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.create(config_dict))
    config: PPOConfig = OmegaConf.to_object(merged)
    config.deep_post_init()

    config.worker.actor.model.model_path = str(merged_model_dir)
    config.worker.actor.model.tokenizer_path = str(merged_model_dir)
    config.worker.actor.model.enable_gradient_checkpointing = False
    config.worker.actor.use_torch_compile = False
    config.worker.actor.fsdp.enable_rank0_init = False
    config.worker.actor.offload.offload_params = False
    config.worker.actor.offload.offload_optimizer = False
    config.worker.rollout.tensor_parallel_size = 1
    config.worker.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.worker.rollout.n = 1
    if args.max_new_tokens is not None:
        config.worker.rollout.response_length = args.max_new_tokens
        config.data.max_response_length = args.max_new_tokens
    return config


def create_dataset(config: PPOConfig, tokenizer, processor) -> RLHFDataset:
    return RLHFDataset(
        data_path=config.data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        image_key=config.data.image_key,
        video_key=config.data.video_key,
        image_dir=config.data.image_dir,
        video_fps=config.data.video_fps,
        max_prompt_length=config.data.max_prompt_length,
        truncation="right",
        format_prompt=config.data.format_prompt,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        filter_overlong_prompts=config.data.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.data.filter_overlong_prompts_workers,
    )


def render_prompt_text(dataset: RLHFDataset, raw_example: dict[str, Any]) -> str:
    example = deepcopy(raw_example)
    messages = dataset._build_messages(example)
    if dataset.image_key in raw_example:
        return dataset.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if dataset.video_key in raw_example:
        return dataset.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return dataset.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_prompt_batch(dataset: RLHFDataset, max_samples: int) -> tuple[DataProto, list[dict[str, Any]], list[dict[str, Any]]]:
    count = min(max_samples, len(dataset))
    if count <= 0:
        raise RuntimeError("Validation dataset is empty.")

    dataset_items = [dataset[index] for index in range(count)]
    raw_examples = [deepcopy(dataset.dataset[index]) for index in range(count)]
    prompt_texts = [render_prompt_text(dataset, raw_example) for raw_example in raw_examples]

    merged_batch = collate_fn(dataset_items)
    tensors = {key: value for key, value in merged_batch.items() if isinstance(value, torch.Tensor)}
    non_tensors = {key: value for key, value in merged_batch.items() if isinstance(value, np.ndarray)}
    non_tensors["prompt_text"] = np.array(prompt_texts, dtype=object)
    non_tensors["sample_index"] = np.array(list(range(count)), dtype=object)

    meta_info = {
        "min_pixels": dataset.min_pixels,
        "max_pixels": dataset.max_pixels,
        "video_fps": dataset.video_fps,
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info), dataset_items, raw_examples


def response_text_from_row(tokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor) -> str:
    valid_length = int(response_mask.sum().item())
    return tokenizer.decode(response_ids[:valid_length], skip_special_tokens=True)


def response_token_texts(tokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor) -> list[str]:
    valid_length = int(response_mask.sum().item())
    return [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in response_ids[:valid_length].tolist()]


def summarize_sample(
    tokenizer,
    sample_index: int,
    prompt_text: str,
    ground_truth: Any,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
    raw_weights: torch.Tensor,
    effective_weights: torch.Tensor,
    valid_flag: bool,
) -> dict[str, Any]:
    response_length = int(response_mask.sum().item())
    response_text = response_text_from_row(tokenizer, response_ids, response_mask)
    token_texts = response_token_texts(tokenizer, response_ids, response_mask)

    tokens = []
    for token_index in range(response_length):
        tokens.append(
            {
                "index": token_index,
                "token_id": int(response_ids[token_index].item()),
                "token_text": token_texts[token_index],
                "is_answer": bool(answer_token_mask[token_index].item() > 0),
                "raw_weight": float(raw_weights[token_index].item()),
                "effective_weight": float(effective_weights[token_index].item()),
            }
        )

    max_weight = max((token["effective_weight"] for token in tokens), default=1.0)
    return {
        "sample_index": sample_index,
        "answer_chain_valid": bool(valid_flag),
        "prompt_text": prompt_text,
        "ground_truth": ground_truth.item() if isinstance(ground_truth, np.generic) else ground_truth,
        "response_text": response_text,
        "response_length": response_length,
        "effective_weight_mean": float(effective_weights[:response_length].mean().item()) if response_length > 0 else 0.0,
        "effective_weight_max": float(max_weight),
        "tokens": tokens,
    }


def token_color(weight: float, max_weight: float, is_answer: bool) -> str:
    if max_weight <= 0:
        strength = 0.0
    else:
        strength = min(max(weight / max_weight, 0.0), 1.0)

    if is_answer:
        r, g, b = 255, 230, 180
    else:
        r = int(255 - 35 * strength)
        g = int(255 - 105 * strength)
        b = int(255 - 185 * strength)
    return f"rgb({r}, {g}, {b})"


def render_html(sample_summaries: list[dict[str, Any]]) -> str:
    sections = []
    for summary in sample_summaries:
        max_weight = max((token["effective_weight"] for token in summary["tokens"]), default=1.0)
        token_spans = []
        for token in summary["tokens"]:
            title = (
                f"idx={token['index']} | "
                f"raw={token['raw_weight']:.4f} | "
                f"effective={token['effective_weight']:.4f}"
            )
            style = (
                f"background:{token_color(token['effective_weight'], max_weight, token['is_answer'])};"
                "padding:2px 3px;margin:1px;border-radius:4px;display:inline-block;"
                "font-family:monospace;white-space:pre-wrap;"
            )
            if token["is_answer"]:
                style += "border:1px solid #d97706;"
            token_spans.append(
                f'<span style="{style}" title="{html.escape(title)}">{html.escape(token["token_text"])}</span>'
            )

        sections.append(
            f"""
            <section class="sample">
              <h2>Sample {summary["sample_index"]}</h2>
              <p><strong>answer_chain_valid</strong>: {summary["answer_chain_valid"]}</p>
              <p><strong>effective_weight_mean</strong>: {summary["effective_weight_mean"]:.4f}</p>
              <p><strong>effective_weight_max</strong>: {summary["effective_weight_max"]:.4f}</p>
              <h3>Prompt</h3>
              <pre>{html.escape(summary["prompt_text"])}</pre>
              <h3>Ground Truth</h3>
              <pre>{html.escape(str(summary["ground_truth"]))}</pre>
              <h3>Response</h3>
              <pre>{html.escape(summary["response_text"])}</pre>
              <h3>Token Weights</h3>
              <div class="tokens">{''.join(token_spans)}</div>
            </section>
            """
        )

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Reasoning Weight Visualization</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.5; }}
        .sample {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
        pre {{ background: #f8f8f8; padding: 12px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
        .tokens {{ line-height: 2.1; }}
      </style>
    </head>
    <body>
      <h1>Reasoning Weight Visualization</h1>
      <p>Hover a token to inspect its raw and effective weights. Orange borders mark answer tokens.</p>
      {''.join(sections)}
    </body>
    </html>
    """


def main() -> None:
    args = parse_args()
    actor_dir, config_path, merged_model_dir = resolve_paths(args.checkpoint)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else actor_dir.parent / "visualization_1gpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path, merged_model_dir, args)
    tokenizer = get_tokenizer(
        config.worker.actor.model.model_path,
        override_chat_template=config.data.override_chat_template,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        config.worker.actor.model.model_path,
        override_chat_template=config.data.override_chat_template,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )

    dataset = create_dataset(config, tokenizer, processor)
    prompts_dp, _, _ = build_prompt_batch(dataset, args.max_samples)

    worker = FSDPWorker(config=config.worker, role="actor_rollout")
    worker.init_model()

    rollout_meta = {
        "min_pixels": config.data.min_pixels,
        "max_pixels": config.data.max_pixels,
        "video_fps": config.data.video_fps,
    }
    rollout_meta.update(dict(config.worker.rollout.val_override_config) if config.worker.rollout.val_override_config else {})
    prompts_dp.meta_info.update(rollout_meta)

    worker.prepare_rollout_engine()
    try:
        generated = worker.generate_sequences(prompts_dp)
    finally:
        worker.release_rollout_engine()

    answer_token_mask = localize_answer_token_batch(
        tokenizer=tokenizer,
        responses=generated.batch["responses"],
        response_mask=generated.batch["response_mask"],
        reward_function_path=config.worker.reward.reward_function,
        skip_special_tokens=config.worker.reward.skip_special_tokens,
    ).token_masks.to(torch.float32)
    generated = generated.union(DataProto.from_dict(tensors={"answer_token_mask": answer_token_mask}))
    generated.meta_info["temperature"] = config.worker.rollout.temperature

    logprob_output = worker.compute_log_probs(generated)
    generated = generated.union(logprob_output)

    raw_weights = generated.batch["token_loss_weights"].detach().cpu()
    response_mask = generated.batch["response_mask"].detach().cpu()
    responses = generated.batch["responses"].detach().cpu()
    answer_token_mask = generated.batch["answer_token_mask"].detach().cpu()
    valid_mask = generated.batch["answer_chain_valid_mask"].detach().cpu() if "answer_chain_valid_mask" in generated.batch else None
    effective_weights = build_effective_token_loss_weights(
        token_loss_weights=raw_weights,
        response_mask=response_mask,
        sequence_weight_mask=valid_mask,
        clip_min=config.worker.actor.reasoning_loss_weight_clip_min,
        clip_max=config.worker.actor.reasoning_loss_weight_clip_max,
        dtype=torch.float32,
    ).detach().cpu()

    sample_summaries = []
    for sample_index in range(len(prompts_dp)):
        prompt_text = str(prompts_dp.non_tensor_batch["prompt_text"][sample_index])
        ground_truth = prompts_dp.non_tensor_batch["ground_truth"][sample_index]
        valid_flag = bool(valid_mask[sample_index].item()) if valid_mask is not None else True
        sample_summaries.append(
            summarize_sample(
                tokenizer=tokenizer,
                sample_index=sample_index,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                response_ids=responses[sample_index],
                response_mask=response_mask[sample_index],
                answer_token_mask=answer_token_mask[sample_index],
                raw_weights=raw_weights[sample_index],
                effective_weights=effective_weights[sample_index],
                valid_flag=valid_flag,
            )
        )

    jsonl_path = output_dir / "reasoning_weight_visualization.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for summary in sample_summaries:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    html_path = output_dir / "reasoning_weight_visualization.html"
    html_path.write_text(render_html(sample_summaries), encoding="utf-8")

    print(f"Wrote HTML visualization to {html_path}")
    print(f"Wrote JSONL data to {jsonl_path}")


if __name__ == "__main__":
    main()
