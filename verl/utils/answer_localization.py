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

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import torch
from transformers import PreTrainedTokenizer

from .py_functional import get_abs_path


@dataclass(frozen=True)
class AnswerSpan:
    answer_text: str
    char_start: int
    char_end: int
    rule: str


@dataclass(frozen=True)
class AnswerTokenLocalization:
    response_text: str
    answer_span: AnswerSpan
    token_mask: torch.Tensor


@dataclass(frozen=True)
class BatchAnswerTokenLocalization:
    response_texts: list[str]
    answer_spans: list[AnswerSpan]
    token_masks: torch.Tensor


def _resolve_reward_function_path(reward_function_path: str) -> str:
    path = reward_function_path.rsplit(":", maxsplit=1)[0]
    abs_path = get_abs_path(path, prompt="Reward function")
    if abs_path is None:
        raise FileNotFoundError(f"Reward function file {reward_function_path} not found.")

    return abs_path


@lru_cache(maxsize=None)
def _load_reward_module(reward_function_path: str) -> ModuleType:
    abs_path = _resolve_reward_function_path(reward_function_path)
    module_name = f"answer_localization_reward_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reward function module from {abs_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _trimmed_subspan(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw_text = text[start:end]
    leading = len(raw_text) - len(raw_text.lstrip())
    trimmed_end = len(raw_text.rstrip())
    trimmed_text = raw_text[leading:trimmed_end]
    if not trimmed_text:
        raise ValueError("Extracted answer span is empty after trimming.")

    return trimmed_text, start + leading, start + trimmed_end


def _find_last_boxed_content_span(response: str) -> tuple[int, int]:
    marker = r"\boxed{"
    start = response.rfind(marker)
    if start < 0:
        raise ValueError("No \\boxed{...} answer found in response.")

    content_start = start + len(marker)
    depth = 0
    content_end = -1
    for index in range(content_start, len(response)):
        char = response[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == -1:
                content_end = index
                break

    if content_end < 0:
        raise ValueError("The last \\boxed{...} answer is not balanced.")

    _, char_start, char_end = _trimmed_subspan(response, content_start, content_end)
    return char_start, char_end


def _extract_math_answer_span(response: str, module: ModuleType) -> AnswerSpan:
    normalized_response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    answer_text = module.extract_boxed_content(normalized_response)
    if answer_text == "None":
        raise ValueError("No valid \\boxed{...} answer found after applying the math reward preprocessing rule.")

    char_start, char_end = _find_last_boxed_content_span(response)
    return AnswerSpan(answer_text=answer_text, char_start=char_start, char_end=char_end, rule="math")


def _extract_r1v_answer_span(response: str) -> AnswerSpan:
    match = re.search(r"<answer>(.*?)</answer>", response)
    if match is not None:
        answer_text, char_start, char_end = _trimmed_subspan(response, *match.span(1))
        return AnswerSpan(answer_text=answer_text, char_start=char_start, char_end=char_end, rule="r1v")

    stripped = response.strip()
    if not stripped:
        raise ValueError("Response is empty after applying the r1v fallback extraction rule.")

    leading = len(response) - len(response.lstrip())
    trailing = len(response.rstrip())
    return AnswerSpan(answer_text=stripped, char_start=leading, char_end=trailing, rule="r1v")


def _extract_dapo_answer_span(response: str) -> AnswerSpan:
    window_start = max(0, len(response) - 300)
    window_text = response[window_start:]
    matches = list(re.finditer(r"(?i)Answer\s*:\s*([^\n]+)", window_text))
    if not matches:
        raise ValueError("No Answer: ... line found in the last 300 characters of the response.")

    match = matches[-1]
    group_start, group_end = match.span(1)
    return AnswerSpan(
        answer_text=window_text[group_start:group_end],
        char_start=window_start + group_start,
        char_end=window_start + group_end,
        rule="dapo",
    )


def _extract_number_game_answer_span(response: str) -> AnswerSpan:
    stripped = response.strip()
    if stripped in {"0", "1", "2"}:
        leading = len(response) - len(response.lstrip())
        return AnswerSpan(answer_text=stripped, char_start=leading, char_end=leading + 1, rule="number_game")

    match = re.search(r"[012]", response)
    if match is None:
        raise ValueError("No 0/1/2 answer found in response.")

    return AnswerSpan(
        answer_text=match.group(0),
        char_start=match.start(),
        char_end=match.end(),
        rule="number_game",
    )


def extract_answer_span(response: str, reward_function_path: str) -> AnswerSpan:
    module = _load_reward_module(reward_function_path)
    reward_name = getattr(module, "REWARD_NAME", os.path.splitext(os.path.basename(module.__file__))[0])

    if reward_name == "math":
        return _extract_math_answer_span(response, module)
    if reward_name == "r1v":
        return _extract_r1v_answer_span(response)
    if reward_name == "dapo":
        return _extract_dapo_answer_span(response)
    if reward_name in {"number_game", "android_gui"}:
        return _extract_number_game_answer_span(response)

    raise NotImplementedError(f"Answer extraction is not implemented for reward `{reward_name}`.")


def _decode_response_prefix_lengths(
    tokenizer: PreTrainedTokenizer,
    valid_response_ids: torch.Tensor,
    skip_special_tokens: bool,
) -> tuple[str, list[int]]:
    prefix_lengths = [0]
    previous_length = 0
    for token_count in range(1, valid_response_ids.size(0) + 1):
        prefix_text = tokenizer.decode(valid_response_ids[:token_count], skip_special_tokens=skip_special_tokens)
        current_length = len(prefix_text)
        if current_length < previous_length:
            raise ValueError("Decoded response prefix lengths must be monotonic.")

        prefix_lengths.append(current_length)
        previous_length = current_length

    response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=skip_special_tokens)
    if prefix_lengths[-1] != len(response_text):
        raise ValueError("Decoded response prefix lengths do not match the full decoded response.")

    return response_text, prefix_lengths


def localize_answer_tokens(
    tokenizer: PreTrainedTokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    reward_function_path: str,
    skip_special_tokens: bool = True,
) -> AnswerTokenLocalization:
    if response_ids.dim() != 1 or response_mask.dim() != 1:
        raise ValueError("response_ids and response_mask must be 1D tensors.")
    if response_ids.size(0) != response_mask.size(0):
        raise ValueError("response_ids and response_mask must have the same length.")

    response_length = int(response_mask.sum().item())
    if response_length <= 0:
        raise ValueError("Response mask does not contain any valid tokens.")

    valid_response_ids = response_ids[:response_length]
    response_text, prefix_lengths = _decode_response_prefix_lengths(
        tokenizer=tokenizer,
        valid_response_ids=valid_response_ids,
        skip_special_tokens=skip_special_tokens,
    )
    answer_span = extract_answer_span(response_text, reward_function_path=reward_function_path)

    if not (0 <= answer_span.char_start < answer_span.char_end <= len(response_text)):
        raise ValueError("Answer span is outside the decoded response text.")

    token_mask = torch.zeros_like(response_mask, dtype=torch.float32)
    for token_index in range(response_length):
        token_start = prefix_lengths[token_index]
        token_end = prefix_lengths[token_index + 1]
        if token_end <= token_start:
            continue

        if token_start < answer_span.char_end and token_end > answer_span.char_start:
            token_mask[token_index] = 1.0

    if token_mask[:response_length].sum().item() <= 0:
        raise ValueError("The extracted answer span does not overlap with any decoded response token.")

    return AnswerTokenLocalization(response_text=response_text, answer_span=answer_span, token_mask=token_mask)


def _localize_answer_tokens_or_empty(
    tokenizer: PreTrainedTokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    reward_function_path: str,
    skip_special_tokens: bool = True,
) -> AnswerTokenLocalization:
    try:
        return localize_answer_tokens(
            tokenizer=tokenizer,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_function_path=reward_function_path,
            skip_special_tokens=skip_special_tokens,
        )
    except ValueError:
        response_length = int(response_mask.sum().item())
        valid_response_ids = response_ids[:response_length]
        response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=skip_special_tokens)
        token_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        return AnswerTokenLocalization(
            response_text=response_text,
            answer_span=AnswerSpan(answer_text="", char_start=0, char_end=0, rule="unlocalized"),
            token_mask=token_mask,
        )


def localize_answer_token_batch(
    tokenizer: PreTrainedTokenizer,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    reward_function_path: str,
    skip_special_tokens: bool = True,
) -> BatchAnswerTokenLocalization:
    if responses.dim() != 2 or response_mask.dim() != 2:
        raise ValueError("responses and response_mask must be 2D tensors.")
    if responses.shape != response_mask.shape:
        raise ValueError("responses and response_mask must have the same shape.")

    localizations = [
        _localize_answer_tokens_or_empty(
            tokenizer=tokenizer,
            response_ids=responses[index],
            response_mask=response_mask[index],
            reward_function_path=reward_function_path,
            skip_special_tokens=skip_special_tokens,
        )
        for index in range(responses.size(0))
    ]
    token_masks = torch.stack([localization.token_mask for localization in localizations], dim=0)
    return BatchAnswerTokenLocalization(
        response_texts=[localization.response_text for localization in localizations],
        answer_spans=[localization.answer_span for localization in localizations],
        token_masks=token_masks,
    )