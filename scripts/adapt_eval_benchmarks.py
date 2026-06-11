#!/usr/bin/env python3

import argparse
import ast
import base64
import csv
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset


DEFAULT_INPUT_ROOT = "datasets/raw_benchmarks"
DEFAULT_OUTPUT_ROOT = "datasets/easyr1_eval_benchmarks"
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def format_options(options: Any) -> str:
    if options is None or options == "":
        return ""
    if isinstance(options, str):
        text = options.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                options = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text
        else:
            return text
    if isinstance(options, dict):
        options = [f"{key}. {value}" for key, value in options.items()]
    if not isinstance(options, (list, tuple)):
        return normalize_text(options)

    lines = []
    for index, option in enumerate(options):
        letter = CHOICE_LETTERS[index] if index < len(CHOICE_LETTERS) else str(index + 1)
        text = normalize_text(option)
        if text.startswith(f"{letter}.") or text.startswith(f"{letter}:") or text.startswith(f"({letter})"):
            lines.append(text)
        else:
            lines.append(f"({letter}) {text}")
    return "\n".join(lines)


def build_prompt(question: Any, options: Any = None, instruction: str | None = None) -> str:
    prompt = normalize_text(question)
    option_text = format_options(options)
    if option_text:
        prompt = f"{prompt}\nChoices:\n{option_text}"
    if instruction:
        prompt = f"{instruction.strip()}\n{prompt}"
    return prompt


def image_count(prompt: str) -> int:
    return prompt.count("<image>")


def reconcile_prompt_with_images(prompt: str, images: list[Any]) -> tuple[str | None, str | None]:
    num_images = len(images)
    placeholders = image_count(prompt)
    if num_images == 0:
        return prompt, None
    if placeholders == num_images:
        return prompt, None
    if num_images == 1 and placeholders == 0:
        return f"<image> {prompt}" if prompt else "<image>", None
    if placeholders == 0:
        return f"{'<image> ' * num_images}{prompt}".strip(), None
    return None, f"images={num_images},placeholders={placeholders}"


def existing_path(root: Path, raw_path: Any, prefixes: tuple[str, ...] = ()) -> str | None:
    raw = normalize_text(raw_path)
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(root / path)
        for prefix in prefixes:
            candidates.append(root / prefix / path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def normalize_image_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("bytes") or value.get("path"):
            return value
        return None
    return value


def finalize_records(records: list[dict[str, Any]], benchmark: str) -> Dataset:
    bad = [record for record in records if record.get("_drop_reason")]
    if bad:
        reasons = Counter(record["_drop_reason"] for record in bad)
        print(f"\nDropped {len(bad)} invalid {benchmark} rows:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    clean = [{key: value for key, value in record.items() if key != "_drop_reason"} for record in records if not record.get("_drop_reason")]
    features = Features(
        {
            "id": Value("string"),
            "prompt": Value("string"),
            "answer": Value("string"),
            "images": Sequence(Image()),
            "benchmark": Value("string"),
            "question_type": Value("string"),
            "answer_type": Value("string"),
            "category": Value("string"),
        }
    )
    return Dataset.from_list(clean, features=features)


def make_record(
    *,
    benchmark: str,
    record_id: Any,
    prompt: str,
    answer: Any,
    images: list[Any],
    question_type: Any = "",
    answer_type: Any = "",
    category: Any = "",
) -> dict[str, Any]:
    images = [image for image in (normalize_image_value(image) for image in images) if image is not None]
    prompt, drop_reason = reconcile_prompt_with_images(prompt, images)
    return {
        "id": normalize_text(record_id),
        "prompt": prompt or "",
        "answer": normalize_text(answer),
        "images": images,
        "benchmark": benchmark,
        "question_type": normalize_text(question_type),
        "answer_type": normalize_text(answer_type),
        "category": normalize_text(category),
        "_drop_reason": drop_reason or "",
    }


def load_parquet_dir(path: Path) -> Dataset:
    files = sorted(str(file) for file in path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return load_dataset("parquet", data_files=files, split="train")


def adapt_mathvista(root: Path) -> Dataset:
    dataset = load_parquet_dir(root / "data")
    records = []
    for example in dataset:
        image = existing_path(root, example.get("image"), prefixes=("images",))
        prompt = build_prompt(example.get("query") or example.get("question"), instruction=None)
        metadata = example.get("metadata") or {}
        records.append(
            make_record(
                benchmark="MathVista",
                record_id=example.get("pid"),
                prompt=prompt,
                answer=example.get("answer"),
                images=[image or example.get("decoded_image")],
                question_type=example.get("question_type"),
                answer_type=example.get("answer_type"),
                category=metadata.get("category", ""),
            )
        )
    return finalize_records(records, "MathVista")


def adapt_mathverse(root: Path) -> Dataset:
    json_file = root / "testmini.json"
    if json_file.is_file():
        examples = json.loads(json_file.read_text(encoding="utf-8"))
    else:
        examples = list(load_dataset("parquet", data_files=str(root / "testmini.parquet"), split="train"))

    records = []
    for example in examples:
        image = existing_path(root, example.get("image"), prefixes=("images", "images/images"))
        metadata = example.get("metadata") or {}
        records.append(
            make_record(
                benchmark="MathVerse",
                record_id=example.get("sample_index"),
                prompt=build_prompt(example.get("query_cot") or example.get("question")),
                answer=example.get("answer"),
                images=[image or example.get("image")],
                question_type=example.get("question_type"),
                answer_type="",
                category=metadata.get("subject", ""),
            )
        )
    return finalize_records(records, "MathVerse")


def adapt_mathvision(root: Path) -> Dataset:
    dataset = load_parquet_dir(root / "data")
    records = []
    for example in dataset:
        image = existing_path(root, example.get("image"), prefixes=("images",))
        records.append(
            make_record(
                benchmark="MathVision",
                record_id=example.get("id"),
                prompt=build_prompt(example.get("question"), example.get("options")),
                answer=example.get("answer"),
                images=[image or example.get("decoded_image")],
                question_type="multiple_choice" if example.get("options") else "free_form",
                answer_type="",
                category=example.get("subject") or example.get("category", ""),
            )
        )
    return finalize_records(records, "MathVision")


def adapt_wemath(root: Path) -> Dataset:
    json_file = root / "testmini.json"
    examples = json.loads(json_file.read_text(encoding="utf-8"))
    records = []
    for example in examples:
        image = existing_path(root, example.get("image_path"), prefixes=("testmini/data", "data"))
        records.append(
            make_record(
                benchmark="WeMath",
                record_id=example.get("ID"),
                prompt=build_prompt(example.get("question"), example.get("option")),
                answer=example.get("answer"),
                images=[image or example.get("image_path")],
                question_type="multiple_choice",
                answer_type="",
                category=example.get("knowledge concept", ""),
            )
        )
    return finalize_records(records, "WeMath")


def adapt_dynamath(root: Path) -> Dataset:
    dataset = load_parquet_dir(root / "data")

    records = []
    for example in dataset:
        image = existing_path(root, example.get("image"), prefixes=("10VariationSamples", "10VariationSamples/10VariationSamples"))
        records.append(
            make_record(
                benchmark="DynaMath",
                record_id=example.get("id"),
                prompt=build_prompt(example.get("question")),
                answer=example.get("ground_truth"),
                images=[image or example.get("decoded_image")],
                question_type="multiple_choice" if example.get("answer_type") == "multiple choice" else "free_form",
                answer_type=example.get("answer_type"),
                category=example.get("subject", ""),
            )
        )
    return finalize_records(records, "DynaMath")


def adapt_logicvista(root: Path) -> Dataset:
    dataset = load_parquet_dir(root / "data")
    records = []
    for example in dataset:
        records.append(
            make_record(
                benchmark="LogicVista",
                record_id=example.get("id"),
                prompt=build_prompt(example.get("question")),
                answer=example.get("answer"),
                images=[example.get("image")],
                question_type="multiple_choice",
                answer_type="text",
                category=", ".join(example.get("broad_capability") or []),
            )
        )
    return finalize_records(records, "LogicVista")


def adapt_mmmu_pro(root: Path, config_name: str = "standard (4 options)") -> Dataset:
    config_root = root / config_name
    dataset = load_parquet_dir(config_root)
    records = []
    for example in dataset:
        images = []
        prompt = normalize_text(example.get("question"))
        for index in range(1, 8):
            key = f"image_{index}"
            image = normalize_image_value(example.get(key))
            if image is not None:
                images.append(image)
                prompt = prompt.replace(f"<image {index}>", "<image>")

        records.append(
            make_record(
                benchmark="MMMU-Pro",
                record_id=example.get("id"),
                prompt=build_prompt(prompt, example.get("options")),
                answer=example.get("answer"),
                images=images,
                question_type="multiple_choice",
                answer_type="letter",
                category=example.get("subject", ""),
            )
        )
    return finalize_records(records, "MMMU-Pro")


def decode_mmstar_image(image_b64: str, output_dir: Path, record_id: str) -> str:
    from PIL import Image as PILImage

    output_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(image_b64)
    image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
    image_path = output_dir / f"{record_id}.jpg"
    image.save(image_path, format="JPEG", quality=95)
    return str(image_path)


def adapt_mmstar(root: Path, output_root: Path) -> Dataset:
    allow_large_csv_fields()
    tsv_file = root / "MMStar.tsv"
    image_output_dir = output_root / "_images" / "MMStar"
    records = []
    with tsv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            record_id = normalize_text(row.get("index"))
            image = decode_mmstar_image(normalize_text(row.get("image")), image_output_dir, record_id)
            records.append(
                make_record(
                    benchmark="MMStar",
                    record_id=record_id,
                    prompt=build_prompt(row.get("question")),
                    answer=row.get("answer"),
                    images=[image],
                    question_type="multiple_choice",
                    answer_type="letter",
                    category=row.get("category", ""),
                )
            )
    return finalize_records(records, "MMStar")


ADAPTERS: dict[str, Callable[..., Dataset]] = {
    "MathVista": adapt_mathvista,
    "MathVerse": adapt_mathverse,
    "MathVision": adapt_mathvision,
    "WeMath": adapt_wemath,
    "DynaMath": adapt_dynamath,
    "LogicVista": adapt_logicvista,
    "MMMU-Pro": adapt_mmmu_pro,
    "MMStar": adapt_mmstar,
}


def write_dataset(dataset: Dataset, output_file: Path, overwrite: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}")
    dataset.to_parquet(str(output_file))
    print(f"Wrote {len(dataset)} rows to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt downloaded eval benchmarks into EasyR1 VLM parquet format.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Root containing raw benchmark directories.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root for adapted EasyR1 parquet files.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(ADAPTERS.keys()),
        choices=list(ADAPTERS.keys()),
        help="Benchmarks to adapt.",
    )
    parser.add_argument(
        "--mmmu-pro-config",
        default="standard (4 options)",
        choices=["standard (4 options)", "standard (10 options)", "vision"],
        help="MMMU-Pro config to adapt.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquet files.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    for benchmark in args.benchmarks:
        print(f"\nAdapting {benchmark}...")
        benchmark_root = input_root / benchmark
        if not benchmark_root.exists():
            raise FileNotFoundError(f"Missing raw benchmark directory: {benchmark_root}")

        if benchmark == "MMMU-Pro":
            dataset = adapt_mmmu_pro(benchmark_root, config_name=args.mmmu_pro_config)
        elif benchmark == "MMStar":
            dataset = adapt_mmstar(benchmark_root, output_root)
        else:
            dataset = ADAPTERS[benchmark](benchmark_root)

        write_dataset(dataset, output_root / benchmark / "test.parquet", overwrite=args.overwrite)

    print("\nEasyR1 config suggestions:")
    print(f"  data.val_files={output_root}/<Benchmark>/test.parquet")
    print("  data.prompt_key=prompt")
    print("  data.answer_key=answer")
    print("  data.image_key=images")
    print("  data.format_prompt=null")


if __name__ == "__main__":
    main()
