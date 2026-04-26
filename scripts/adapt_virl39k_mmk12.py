#!/usr/bin/env python3

import argparse
import math
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset


def _filter_overlong(
    dataset: Dataset,
    model_path: str,
    max_prompt_length: int,
    min_pixels: int,
    max_pixels: int,
    num_proc: int,
) -> Dataset:
    from PIL import Image as PILImage
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)

    def _is_short_enough(example: dict[str, Any]) -> bool:
        images = example.get("images") or []
        prompt = example.get("prompt", "")
        content = []
        for index, text_part in enumerate(prompt.split("<image>")):
            if index != 0:
                content.append({"type": "image"})
            if text_part:
                content.append({"type": "text", "text": text_part})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        pil_images = []
        for img in images:
            if isinstance(img, dict) and img.get("bytes"):
                pil_images.append(PILImage.open(BytesIO(img["bytes"])).convert("RGB"))
            elif isinstance(img, str):
                pil_images.append(PILImage.open(img).convert("RGB"))
            elif hasattr(img, "convert"):
                pil_images.append(img.convert("RGB"))

        resized = []
        for pil in pil_images:
            if max_pixels and pil.width * pil.height > max_pixels:
                f = math.sqrt(max_pixels / (pil.width * pil.height))
                pil = pil.resize((int(pil.width * f), int(pil.height * f)))
            if min_pixels and pil.width * pil.height < min_pixels:
                f = math.sqrt(min_pixels / (pil.width * pil.height))
                pil = pil.resize((int(pil.width * f), int(pil.height * f)))
            resized.append(pil)

        if resized:
            inputs = processor(resized, [text], add_special_tokens=False, return_tensors="pt")
        else:
            inputs = processor(text=[text], add_special_tokens=False, return_tensors="pt")
        return inputs["input_ids"].size(-1) <= max_prompt_length

    before = len(dataset)
    dataset = dataset.filter(_is_short_enough, desc="Filtering overlong prompts", num_proc=num_proc)
    print(f"Filtered {before - len(dataset)} overlong samples, {len(dataset)} remaining.")
    return dataset


DEFAULT_TRAIN_DIR = "/projects_vol/gp_boan/easyr1_assets/datasets/train/ViRL39K"
DEFAULT_VAL_DIR = "/projects_vol/gp_boan/easyr1_assets/datasets/val/MMK12/data"
DEFAULT_OUTPUT_ROOT = "/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"
DEFAULT_VAL_GLOB = "test-*.parquet"


def normalize_prompt(question: Any) -> str:
    return str(question).strip()


def reconcile_prompt_with_images(prompt: str, num_images: int) -> tuple[Optional[str], Optional[str]]:
    placeholder_count = prompt.count("<image>")

    if num_images == 0:
        return prompt, None

    if placeholder_count == num_images:
        return prompt, None

    if num_images == 1 and placeholder_count == 0:
        fixed_prompt = f"<image> {prompt}" if prompt else "<image>"
        return fixed_prompt, None

    return None, f"images={num_images},placeholders={placeholder_count}"


def maybe_wrap_unboxed_answer(answer: Any, wrap_unboxed: bool) -> str:
    text = str(answer).strip()
    if not wrap_unboxed or not text or "\\boxed{" in text:
        return text
    return f"\\boxed{{{text}}}"


def resolve_record_id(example: dict[str, Any]) -> str | None:
    for key in ("qid", "id"):
        value = example.get(key)
        if value is not None:
            return str(value)
    return None


def normalize_train_images(image_value: Any, train_dir: Path) -> list[Any]:
    if image_value is None:
        return []
    if not isinstance(image_value, list):
        image_value = list(image_value) if isinstance(image_value, tuple) else [image_value]

    images: list[Any] = []
    for item in image_value:
        if isinstance(item, str):
            image_path = Path(item)
            if not image_path.is_absolute():
                candidate = train_dir / image_path
                if not candidate.exists():
                    alt = train_dir / "images" / image_path
                    if alt.exists():
                        candidate = alt
                image_path = candidate
            images.append(str(image_path))
        else:
            images.append(item)
    return images


def normalize_val_images(image_value: Any) -> list[Any]:
    if image_value is None:
        return []
    if isinstance(image_value, list):
        return image_value
    if isinstance(image_value, tuple):
        return list(image_value)
    return [image_value]


def finalize_dataset(dataset: Dataset, features: Features, split_name: str) -> Dataset:
    invalid_rows = dataset.filter(lambda example: example["_drop_reason"] != "", desc=f"Collecting bad {split_name} rows")
    if len(invalid_rows) > 0:
        reasons = Counter(invalid_rows["_drop_reason"])
        print(f"\nDropped {len(invalid_rows)} invalid {split_name} rows:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    dataset = dataset.filter(lambda example: example["_drop_reason"] == "", desc=f"Filtering bad {split_name} rows")
    dataset = dataset.remove_columns(["_drop_reason"])
    return dataset.cast(features)


def build_output_features(
    include_category: bool, include_source: bool, include_pass_rates: bool, embed_images: bool = False
) -> Features:
    image_feature = Image() if embed_images else Value("string")
    features_dict: dict[str, Any] = {
        "id": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
        "images": Sequence(image_feature),
    }
    if include_category:
        features_dict["category"] = Value("string")
    if include_source:
        features_dict["source"] = Value("string")
    if include_pass_rates:
        features_dict["PassRate_32BTrained"] = Value("float64")
        features_dict["PassRate_7BBase"] = Value("float64")
    return Features(features_dict)


def adapt_virl39k_train(train_dir: Path) -> Dataset:
    train_file = train_dir / "39Krelease.parquet"
    if not train_file.is_file():
        raise FileNotFoundError(f"ViRL39K parquet not found: {train_file}")

    dataset = load_dataset("parquet", data_files=str(train_file), split="train")
    features = build_output_features(
        include_category="category" in dataset.column_names,
        include_source="source" in dataset.column_names,
        include_pass_rates=(
            "PassRate_32BTrained" in dataset.column_names and "PassRate_7BBase" in dataset.column_names
        ),
    )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        prompt = normalize_prompt(example["question"])
        images = normalize_train_images(example["image"], train_dir)
        prompt, drop_reason = reconcile_prompt_with_images(prompt, len(images))
        output = {
            "id": resolve_record_id(example) or "",
            "prompt": prompt or "",
            "answer": str(example["answer"]).strip(),
            "images": images,
            "_drop_reason": drop_reason or "",
        }
        if "category" in features:
            output["category"] = str(example.get("category", ""))
        if "source" in features:
            output["source"] = str(example.get("source", ""))
        if "PassRate_32BTrained" in features:
            output["PassRate_32BTrained"] = float(example.get("PassRate_32BTrained", -1.0))
            output["PassRate_7BBase"] = float(example.get("PassRate_7BBase", -1.0))
        return output

    dataset = dataset.map(
        _convert,
        remove_columns=dataset.column_names,
        desc="Adapting ViRL39K train set",
    )
    return finalize_dataset(dataset, features, "train")


def adapt_mmk12_val(val_dir: Path, val_glob: str, wrap_unboxed_answers: bool) -> Dataset:
    val_files = sorted(str(path) for path in val_dir.glob(val_glob))
    if not val_files:
        raise FileNotFoundError(f"No MMK12 parquet files matched {val_glob} under {val_dir}")

    dataset = load_dataset("parquet", data_files=val_files, split="train")
    features = build_output_features(
        include_category="category" in dataset.column_names or "subject" in dataset.column_names,
        include_source=False,
        include_pass_rates=False,
        embed_images=True,
    )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        category = example.get("category", example.get("subject", ""))
        prompt = normalize_prompt(example["question"])
        images = normalize_val_images(example["image"])
        prompt, drop_reason = reconcile_prompt_with_images(prompt, len(images))
        return {
            "id": resolve_record_id(example) or "",
            "prompt": prompt or "",
            "answer": maybe_wrap_unboxed_answer(example["answer"], wrap_unboxed_answers),
            "images": images,
            "_drop_reason": drop_reason or "",
            **({"category": str(category)} if "category" in features else {}),
        }

    dataset = dataset.map(
        _convert,
        remove_columns=dataset.column_names,
        desc="Adapting MMK12 validation set",
    )
    return finalize_dataset(dataset, features, "validation")


def write_dataset(dataset: Dataset, output_file: Path, overwrite: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}")
    dataset.to_parquet(str(output_file))
    print(f"Wrote {len(dataset)} rows to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt ViRL39K/MMK12 datasets into EasyR1 standard format.")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Directory containing ViRL39K data.")
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR, help="Directory containing MMK12 parquet shards.")
    parser.add_argument("--val-glob", default=DEFAULT_VAL_GLOB, help="Glob for MMK12 validation shards.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory. Train/val parquet files will be written under this directory.",
    )
    parser.add_argument(
        "--wrap-unboxed-val-answers",
        action="store_true",
        help="Wrap MMK12 validation answers in \\boxed{} when they are not already boxed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files.",
    )
    parser.add_argument("--model-path", default=None, help="Model path for offline overlong-prompt filtering.")
    parser.add_argument("--max-prompt-length", type=int, default=2048, help="Max prompt token length for filtering.")
    parser.add_argument("--min-pixels", type=int, default=262144, help="Min pixels for image resize during filtering.")
    parser.add_argument("--max-pixels", type=int, default=4194304, help="Max pixels for image resize during filtering.")
    parser.add_argument("--filter-workers", type=int, default=4, help="Number of workers for filtering.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_output = output_root / "train" / "part-00000.parquet"
    val_output = output_root / "val" / "part-00000.parquet"

    train_dataset = adapt_virl39k_train(Path(args.train_dir))
    if args.model_path:
        print(f"\nFiltering train set (max_prompt_length={args.max_prompt_length}, max_pixels={args.max_pixels})...")
        train_dataset = _filter_overlong(
            train_dataset, args.model_path, args.max_prompt_length,
            args.min_pixels, args.max_pixels, args.filter_workers,
        )
    val_dataset = adapt_mmk12_val(
        Path(args.val_dir),
        args.val_glob,
        wrap_unboxed_answers=args.wrap_unboxed_val_answers,
    )

    write_dataset(train_dataset, train_output, overwrite=args.overwrite)
    write_dataset(val_dataset, val_output, overwrite=args.overwrite)

    print("\nEasyR1 config suggestions:")
    print(f"  data.train_files={train_output.parent}")
    print(f"  data.val_files={val_output.parent}")
    print("  data.prompt_key=prompt")
    print("  data.answer_key=answer")
    print("  data.image_key=images")
    if args.model_path:
        print("  data.filter_overlong_prompts=false  # already filtered offline")
    else:
        print("  data.filter_overlong_prompts=true")


if __name__ == "__main__":
    main()
