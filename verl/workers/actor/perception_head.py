import math

import torch
from torch import nn


def resolve_model_hidden_size(config) -> int:
    for candidate in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ):
        hidden_size = getattr(candidate, "hidden_size", None) if candidate is not None else None
        if hidden_size is not None:
            return int(hidden_size)
    raise RuntimeError("Unable to resolve hidden_size for the perception head.")


def resolve_visual_tower_module(module: nn.Module):
    for candidate in (
        module,
        getattr(module, "model", None),
        getattr(module, "_fsdp_wrapped_module", None),
        getattr(getattr(module, "_fsdp_wrapped_module", None), "model", None),
    ):
        if candidate is None:
            continue
        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual
        nested_model = getattr(candidate, "model", None)
        if nested_model is not None:
            visual = getattr(nested_model, "visual", None)
            if visual is not None:
                return visual
    return None


def mark_visual_tower_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        setattr(parameter, "_is_visual_tower_param", True)


def is_visual_tower_parameter(parameter: torch.nn.Parameter) -> bool:
    return bool(getattr(parameter, "_is_visual_tower_param", False))


def mark_perception_head_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        setattr(parameter, "_is_perception_head_param", True)


def is_perception_head_parameter(parameter: torch.nn.Parameter) -> bool:
    return bool(getattr(parameter, "_is_perception_head_param", False))


class PerceptionEvidenceHead(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int | None = None):
        super().__init__()
        projection_size = hidden_size if projection_size is None or projection_size <= 0 else projection_size
        self.visual_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.score_proj = nn.Linear(projection_size, 1, bias=False)
        self.scale = 1.0 / math.sqrt(projection_size)

    def forward(self, visual_states: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.visual_proj.weight.dtype
        projected = self.visual_proj(visual_states.to(dtype=weight_dtype))
        return self.score_proj(projected).squeeze(-1) * self.scale
