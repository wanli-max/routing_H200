from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchAnswerChainSupport:
    token_loss_weights: torch.Tensor
    answer_chain_valid_mask: torch.Tensor
    visual_target: torch.Tensor | None


def compute_routable_reasoning_sequence_mask(
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
) -> torch.Tensor:
    if response_mask.dim() != 2 or answer_token_mask.dim() != 2:
        raise ValueError("response_mask and answer_token_mask must be rank-2 tensors.")
    if response_mask.shape != answer_token_mask.shape:
        raise ValueError("response_mask and answer_token_mask must have identical shapes.")

    current_answer_mask = answer_token_mask.to(torch.bool) & response_mask.to(torch.bool)
    answer_present = current_answer_mask.any(dim=-1)
    first_answer_index = torch.argmax(current_answer_mask.to(torch.int64), dim=-1)
    token_positions = torch.arange(response_mask.size(1), device=response_mask.device).unsqueeze(0)
    reasoning_region = token_positions < first_answer_index.unsqueeze(1)
    reasoning_token_count = (reasoning_region & response_mask.to(torch.bool)).sum(dim=-1)
    return answer_present & (reasoning_token_count > 0)


def build_trivial_answer_chain_support(
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
    produce_visual_target: bool = True,
) -> BatchAnswerChainSupport:
    if attention_mask.dim() != 2 or response_mask.dim() != 2 or answer_token_mask.dim() != 2:
        raise ValueError("attention_mask, response_mask, and answer_token_mask must be rank-2 tensors.")
    if response_mask.shape != answer_token_mask.shape:
        raise ValueError("response_mask and answer_token_mask must have identical shapes.")
    if attention_mask.size(0) != response_mask.size(0):
        raise ValueError("attention_mask and response_mask must have identical batch size.")

    dtype = torch.float32
    token_loss_weights = (answer_token_mask.to(torch.bool) & response_mask.to(torch.bool)).to(dtype)
    answer_chain_valid_mask = torch.zeros(response_mask.size(0), device=response_mask.device, dtype=dtype)
    visual_target = (
        torch.zeros((attention_mask.size(0), attention_mask.size(1)), device=attention_mask.device, dtype=dtype)
        if produce_visual_target
        else None
    )
    return BatchAnswerChainSupport(
        token_loss_weights=token_loss_weights.detach(),
        answer_chain_valid_mask=answer_chain_valid_mask.detach(),
        visual_target=None if visual_target is None else visual_target.detach(),
    )


def _compute_response_span(attention_mask: torch.Tensor, response_mask: torch.Tensor) -> tuple[int, int]:
    sequence_width = attention_mask.size(0)
    response_width = response_mask.size(0)
    response_length = int(response_mask.sum().item())
    if int(attention_mask.sum().item()) <= 0:
        raise ValueError("attention_mask does not contain any valid tokens.")
    if response_length <= 0:
        raise ValueError("response_mask does not contain any valid response tokens.")
    if response_width > sequence_width:
        raise ValueError("response width cannot exceed sequence width.")
    if response_length > response_width:
        raise ValueError("response length cannot exceed response width.")

    response_start = sequence_width - response_width
    response_end = response_start + response_length
    return response_start, response_end


def _compute_reasoning_mask(valid_response_mask: torch.Tensor, answer_token_mask: torch.Tensor) -> torch.Tensor:
    answer_indices = torch.nonzero(answer_token_mask > 0, as_tuple=False).flatten()
    reasoning_mask = torch.zeros_like(valid_response_mask, dtype=torch.float32)
    if answer_indices.numel() <= 0:
        return reasoning_mask

    reasoning_end = int(answer_indices[0].item())
    if reasoning_end > 0:
        reasoning_mask[:reasoning_end] = valid_response_mask[:reasoning_end].to(torch.float32)
    return reasoning_mask


def compute_answer_chain_support_from_local_rows(
    predecessor_indices: Sequence[torch.Tensor],
    predecessor_weights: Sequence[torch.Tensor],
    predecessor_valid_masks: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
    visual_token_mask: torch.Tensor | None = None,
    produce_visual_target: bool = True,
    eps_norm: float = 1e-8,
    tiny_threshold: float = 1e-8,
) -> BatchAnswerChainSupport:
    if attention_mask.dim() != 2 or response_mask.dim() != 2 or answer_token_mask.dim() != 2:
        raise ValueError("attention_mask, response_mask, and answer_token_mask must be rank-2 tensors.")
    if response_mask.shape != answer_token_mask.shape:
        raise ValueError("response_mask and answer_token_mask must have identical shapes.")
    batch_size = response_mask.size(0)
    if (
        len(predecessor_indices) != batch_size
        or len(predecessor_weights) != batch_size
        or len(predecessor_valid_masks) != batch_size
    ):
        raise ValueError("Local predecessor rows must provide one padded entry per batch element.")

    response_width = response_mask.size(1)
    sequence_width = attention_mask.size(1)
    device = attention_mask.device
    dtype = torch.float32

    if produce_visual_target and visual_token_mask is None:
        visual_token_mask = torch.zeros_like(attention_mask, dtype=dtype)
    elif produce_visual_target and visual_token_mask.shape != attention_mask.shape:
        raise ValueError("visual_token_mask must match attention_mask shape.")

    token_loss_weights = torch.zeros((batch_size, response_width), device=device, dtype=dtype)
    answer_chain_valid_mask = torch.zeros(batch_size, device=device, dtype=dtype)
    visual_target = torch.zeros((batch_size, sequence_width), device=device, dtype=dtype) if produce_visual_target else None

    for batch_index in range(batch_size):
        current_response_mask = response_mask[batch_index].to(torch.bool)
        current_answer_mask = answer_token_mask[batch_index].to(torch.bool) & current_response_mask
        if current_answer_mask.sum().item() <= 0:
            continue

        response_start, response_end = _compute_response_span(attention_mask[batch_index], response_mask[batch_index])
        response_length = response_end - response_start
        sample_indices = predecessor_indices[batch_index]
        sample_weights = predecessor_weights[batch_index]
        sample_valid = predecessor_valid_masks[batch_index]
        if sample_indices.dim() != 2 or sample_weights.dim() != 2 or sample_valid.dim() != 2:
            raise ValueError("Each batch element must provide rank-2 padded predecessor tensors.")
        if sample_indices.shape != sample_weights.shape or sample_indices.shape != sample_valid.shape:
            raise ValueError("Padded predecessor indices, weights, and valid masks must share the same shape.")
        if sample_indices.size(0) != response_length:
            raise ValueError("Each batch element must provide one predecessor row per valid response token.")

        valid_answer_mask = current_answer_mask[:response_length].to(dtype)
        token_loss_weights[batch_index, :response_length] = valid_answer_mask

        current_reasoning_mask = _compute_reasoning_mask(current_response_mask, current_answer_mask)
        if current_reasoning_mask[:response_length].sum().item() <= 0:
            continue
        reasoning_token_count = current_reasoning_mask[:response_length].sum().clamp_min(1.0).to(dtype)

        full_support = torch.zeros(attention_mask.size(1), device=device, dtype=dtype)
        full_support[response_start:response_end] = valid_answer_mask / valid_answer_mask.sum().clamp_min(1.0)

        for row_offset in range(response_length - 1, -1, -1):
            query_position = response_start + row_offset
            query_mass = full_support[query_position]
            if query_mass <= 0:
                continue

            valid_predecessor_mask = sample_valid[row_offset]
            if valid_predecessor_mask.sum().item() <= 0:
                continue

            row_indices = sample_indices[row_offset][valid_predecessor_mask]
            row_weights = sample_weights[row_offset][valid_predecessor_mask]
            in_range_mask = torch.logical_and(row_indices >= 0, row_indices < response_end)
            row_indices = row_indices[in_range_mask]
            row_weights = row_weights[in_range_mask]
            if row_indices.numel() == 0:
                continue

            row_weight_sum = row_weights.sum()
            if row_weight_sum <= tiny_threshold:
                continue

            full_support[row_indices] += query_mass * (row_weights / (row_weight_sum + eps_norm))

        raw_reasoning_score = full_support[response_start:response_end] * current_reasoning_mask[:response_length].to(dtype)
        score_sum = raw_reasoning_score.sum()
        if score_sum > tiny_threshold and current_reasoning_mask[:response_length].sum().item() > 0:
            answer_chain_valid_mask[batch_index] = 1.0
            token_loss_weights[batch_index, :response_length] += (
                raw_reasoning_score / (score_sum + eps_norm)
            ) * reasoning_token_count

        if visual_target is not None and visual_token_mask is not None:
            current_visual_mass = full_support[:response_end] * visual_token_mask[batch_index, :response_end].to(dtype)
            visual_mass_sum = current_visual_mass.sum()
            if visual_mass_sum > tiny_threshold:
                visual_target[batch_index, :response_end] = current_visual_mass / (visual_mass_sum + eps_norm)

    return BatchAnswerChainSupport(
        token_loss_weights=token_loss_weights.detach(),
        answer_chain_valid_mask=answer_chain_valid_mask.detach(),
        visual_target=None if visual_target is None else visual_target.detach(),
    )
