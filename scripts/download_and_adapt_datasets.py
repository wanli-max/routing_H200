#!/usr/bin/env python3
"""
Download (or adapt from local files) ViRL39K + MMK12 into EasyR1 format.

Usage — from HuggingFace (requires internet):
    python3 scripts/download_and_adapt_datasets.py \
        --output-root /path/to/output/virl39k_mmk12_easyr1 \
        [--hf-cache /path/to/hf/cache] \
        [--overwrite]

Usage — from local parquet files (offline):
    python3 scripts/download_and_adapt_datasets.py \
        --virl39k-path /data/ViRL39K/train \
        --mmk12-path   /data/MMK12/test \
        --output-root  ./datasets/virl39k_mmk12_easyr1 \
        [--overwrite]

  --virl39k-path / --mmk12-path accept either:
    • A directory containing *.parquet files, or
    • A directory saved with dataset.save_to_disk() (Arrow format).

Output structure:
    <output-root>/
        train/part-00000.parquet   # ViRL39K adapted (39k rows, images embedded)
        val/part-00000.parquet     # MMK12 test adapted  (2k rows, images embedded)

Then set in your job script:
    data.train_files=<output-root>/train
    data.val_files=<output-root>/val
    data.prompt_key=prompt
    data.answer_key=answer
    data.image_key=images
"""

import argparse
import glob as _glob
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset


# ── HuggingFace dataset IDs ───────────────────────────────────────────────────
VIRL39K_HF_ID = "TIGER-Lab/ViRL39K"
MMK12_HF_ID = "FanqingM/MMK12"

DEFAULT_OUTPUT_ROOT = "/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"


# ── shared helpers (same logic as adapt_virl39k_mmk12.py) ────────────────────

def normalize_prompt(question: Any) -> str:
    return str(question).strip()


def reconcile_prompt_with_images(prompt: str, num_images: int) -> tuple[Optional[str], Optional[str]]:
    placeholder_count = prompt.count("<image>")
    if num_images == 0:
        return prompt, None
    if placeholder_count == num_images:
        return prompt, None
    if num_images == 1 and placeholder_count == 0:
        return f"<image> {prompt}" if prompt else "<image>", None
    return None, f"images={num_images},placeholders={placeholder_count}"


def maybe_wrap_unboxed_answer(answer: Any, wrap_unboxed: bool) -> str:
    text = str(answer).strip()
    if not wrap_unboxed or not text or "\\boxed{" in text:
        return text
    return f"\\boxed{{{text}}}"


def resolve_record_id(example: dict[str, Any]) -> Optional[str]:
    for key in ("qid", "id"):
        if example.get(key) is not None:
            return str(example[key])
    return None


def normalize_images(image_value: Any) -> list[Any]:
    """Normalise any image representation to a plain list."""
    if image_value is None:
        return []
    if isinstance(image_value, (list, tuple)):
        return list(image_value)
    return [image_value]


def build_output_features(include_category: bool, include_source: bool, include_pass_rates: bool) -> Features:
    return Features({
        "id": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
        "images": Sequence(Image()),
        **( {"category": Value("string")} if include_category else {}),
        **( {"source":   Value("string")} if include_source   else {}),
        **( {"PassRate_32BTrained": Value("float64"),
             "PassRate_7BBase":     Value("float64")} if include_pass_rates else {}),
    })


def finalize_dataset(dataset: Dataset, features: Features, split_name: str) -> Dataset:
    bad = dataset.filter(lambda ex: ex["_drop_reason"] != "", desc=f"Collecting bad {split_name} rows")
    if bad:
        reasons = Counter(bad["_drop_reason"])
        print(f"\nDropped {len(bad)} invalid {split_name} rows:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")
    dataset = dataset.filter(lambda ex: ex["_drop_reason"] == "", desc=f"Filtering bad {split_name} rows")
    dataset = dataset.remove_columns(["_drop_reason"])
    return dataset.cast(features)


# ── local dataset loader ──────────────────────────────────────────────────────

def _disable_image_decoding(ds: Dataset) -> Dataset:
    """Cast all Image columns to Image(decode=False).

    When a parquet file stores images as {"bytes": ..., "path": "foo.jpg"},
    HuggingFace datasets will try to open the path as a local file when
    iterating rows.  Setting decode=False returns the raw dict instead,
    which our _convert functions handle correctly via normalize_images().
    """
    from datasets import Image as HFImage, Sequence as HFSequence
    for col, feature in ds.features.items():
        if isinstance(feature, HFImage):
            ds = ds.cast_column(col, HFImage(decode=False))
        elif isinstance(feature, HFSequence) and isinstance(feature.feature, HFImage):
            ds = ds.cast_column(col, HFSequence(HFImage(decode=False)))
    return ds

def _load_local(path: str, label: str, split_prefix: Optional[str] = None) -> Dataset:
    """Load a dataset from a local directory.

    Supports two layouts:
    1. Directory containing *.parquet files (e.g. downloaded from HF Hub).
    2. Directory saved with dataset.save_to_disk() (Arrow format).

    Args:
        split_prefix: If set, only load parquet files whose name starts with
                      this prefix (e.g. "test-" to select test-*.parquet when
                      train-*.parquet files are present in the same directory).
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{label}: path is not a directory: {p}")

    all_parquet = sorted(_glob.glob(str(p / "*.parquet")))
    if split_prefix:
        parquet_files = [f for f in all_parquet if Path(f).name.startswith(split_prefix)]
        if not parquet_files and all_parquet:
            raise FileNotFoundError(
                f"{label}: no parquet files starting with '{split_prefix}' in {p}. "
                f"Available files: {[Path(f).name for f in all_parquet]}"
            )
    else:
        parquet_files = all_parquet

    if parquet_files:
        print(f"      Loading {len(parquet_files)} parquet file(s) from {p}")
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        return _disable_image_decoding(ds)

    # Fall back to Arrow / save_to_disk layout
    dataset_info = p / "dataset_info.json"
    if dataset_info.exists():
        from datasets import load_from_disk
        print(f"      Loading Arrow dataset from {p}")
        ds = load_from_disk(str(p))
        # load_from_disk may return a DatasetDict; take the first split
        if hasattr(ds, "keys"):
            split_name = next(iter(ds.keys()))
            print(f"      Using split '{split_name}' from DatasetDict")
            ds = ds[split_name]
        return _disable_image_decoding(ds)

    raise FileNotFoundError(
        f"{label}: no *.parquet files and no dataset_info.json found under {p}. "
        "Expected either parquet files or a dataset saved with save_to_disk()."
    )


# ── ViRL39K ───────────────────────────────────────────────────────────────────

def adapt_virl39k_train(hf_cache: Optional[str], local_path: Optional[str] = None) -> Dataset:
    if local_path:
        print(f"[1/2] Loading ViRL39K from local path: {local_path}")
        raw = _load_local(local_path, "ViRL39K")
    else:
        print(f"[1/2] Downloading {VIRL39K_HF_ID} ...")
        kwargs = {"cache_dir": hf_cache} if hf_cache else {}
        raw = load_dataset(VIRL39K_HF_ID, split="train", **kwargs)
    print(f"      {len(raw)} rows loaded.")

    features = build_output_features(
        include_category="category" in raw.column_names,
        include_source="source" in raw.column_names,
        include_pass_rates=(
            "PassRate_32BTrained" in raw.column_names and "PassRate_7BBase" in raw.column_names
        ),
    )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        prompt = normalize_prompt(example["question"])
        images = normalize_images(example.get("image") or example.get("images"))
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

    dataset = raw.map(_convert, remove_columns=raw.column_names, desc="Adapting ViRL39K")
    return finalize_dataset(dataset, features, "train")


# ── MMK12 ─────────────────────────────────────────────────────────────────────

def adapt_mmk12_val(hf_cache: Optional[str], wrap_unboxed: bool, local_path: Optional[str] = None) -> Dataset:
    if local_path:
        print(f"[2/2] Loading MMK12 from local path: {local_path}")
        raw = _load_local(local_path, "MMK12", split_prefix="test-")
    else:
        print(f"[2/2] Downloading {MMK12_HF_ID} (test split) ...")
        kwargs = {"cache_dir": hf_cache} if hf_cache else {}
        raw = load_dataset(MMK12_HF_ID, split="test", **kwargs)
    print(f"      {len(raw)} rows loaded.")

    features = build_output_features(
        include_category=("category" in raw.column_names or "subject" in raw.column_names),
        include_source=False,
        include_pass_rates=False,
    )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        category = example.get("category") or example.get("subject") or ""
        prompt = normalize_prompt(example["question"])
        images = normalize_images(example.get("image") or example.get("images"))
        prompt, drop_reason = reconcile_prompt_with_images(prompt, len(images))
        output = {
            "id": resolve_record_id(example) or "",
            "prompt": prompt or "",
            "answer": maybe_wrap_unboxed_answer(example["answer"], wrap_unboxed),
            "images": images,
            "_drop_reason": drop_reason or "",
        }
        if "category" in features:
            output["category"] = str(category)
        return output

    dataset = raw.map(_convert, remove_columns=raw.column_names, desc="Adapting MMK12")
    return finalize_dataset(dataset, features, "validation")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt ViRL39K + MMK12 into EasyR1 format (from HuggingFace or local files)."
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Output root directory.")
    # ── HuggingFace mode ──────────────────────────────────────────────────────
    parser.add_argument("--hf-cache", default=None,
                        help="HuggingFace cache directory. When set, load from local HF cache "
                             "instead of downloading. Ignored if --virl39k-path is provided.")
    # ── Local file mode ───────────────────────────────────────────────────────
    parser.add_argument("--virl39k-path", default=None,
                        help="Path to local ViRL39K dataset directory (parquet files or "
                             "save_to_disk Arrow format). Overrides HuggingFace download.")
    parser.add_argument("--mmk12-path", default=None,
                        help="Path to local MMK12 dataset directory (parquet files or "
                             "save_to_disk Arrow format). Overrides HuggingFace download.")
    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--wrap-unboxed-val-answers", action="store_true",
                        help="Wrap MMK12 answers in \\boxed{} if not already boxed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    args = parser.parse_args()

    if (args.virl39k_path is None) != (args.mmk12_path is None):
        parser.error("--virl39k-path and --mmk12-path must be provided together.")

    if args.hf_cache and not args.virl39k_path:
        os.environ["HF_HOME"] = args.hf_cache

    output_root = Path(args.output_root)
    train_out = output_root / "train" / "part-00000.parquet"
    val_out   = output_root / "val"   / "part-00000.parquet"

    for path in (train_out, val_out):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}  (use --overwrite to replace)")

    train_ds = adapt_virl39k_train(args.hf_cache, local_path=args.virl39k_path)
    val_ds   = adapt_mmk12_val(args.hf_cache, args.wrap_unboxed_val_answers, local_path=args.mmk12_path)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_ds.to_parquet(str(train_out))
    print(f"Wrote {len(train_ds)} train rows → {train_out}")
    val_ds.to_parquet(str(val_out))
    print(f"Wrote {len(val_ds)} val rows   → {val_out}")

    print("\n✓ Done. Use these in your job script:")
    print(f"  data.train_files={train_out.parent}")
    print(f"  data.val_files={val_out.parent}")
    print("  data.prompt_key=prompt")
    print("  data.answer_key=answer")
    print("  data.image_key=images")
    print("  data.filter_overlong_prompts=false  # already handled at runtime")


if __name__ == "__main__":
    main()
