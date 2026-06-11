#!/usr/bin/env python3
"""Extract validation reward curves from TensorBoard logs.

Run from anywhere inside this repository:

    python scripts/extract_eval_reward_json.py

Default input:
    tensorboard_log/easy_r1/<experiment_name>/

Default output:
    extract_results/<experiment_name>_reward.json

Each output JSON is exactly two lists:
    [[step0, step1, ...], [reward0, reward1, ...]]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TAG = "val/reward_score"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Extract validation reward score curves as small JSON files.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=root / "tensorboard_log" / "easy_r1",
        help="Relative or absolute TensorBoard project log directory. Default: tensorboard_log/easy_r1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "extract_results",
        help="Relative or absolute output directory. Default: extract_results",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"TensorBoard scalar tag to extract. Default: {DEFAULT_TAG}",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for reward values. Default: 6",
    )
    return parser.parse_args()


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit("TensorBoard is required. Run inside the training environment with tensorboard installed.") from exc
    return EventAccumulator


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def find_experiment_dirs(log_root: Path) -> list[Path]:
    log_root = resolve_path(log_root)
    if not log_root.exists():
        raise SystemExit(f"TensorBoard log root does not exist: {log_root}")

    if any(log_root.glob("events.out.tfevents.*")):
        return [log_root]

    experiment_dirs = [
        child
        for child in sorted(log_root.iterdir())
        if child.is_dir() and any(child.rglob("events.out.tfevents.*"))
    ]
    if not experiment_dirs:
        raise SystemExit(f"No experiment event files found under: {log_root}")
    return experiment_dirs


def find_event_dirs(experiment_dir: Path) -> list[Path]:
    event_files = sorted(experiment_dir.rglob("events.out.tfevents.*"))
    return sorted({event_file.parent for event_file in event_files})


def collect_reward_points(experiment_dir: Path, tag: str) -> list[tuple[int, float, float]]:
    EventAccumulator = load_event_accumulator()
    points: list[tuple[int, float, float]] = []

    for event_dir in find_event_dirs(experiment_dir):
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            print(f"[WARN] Failed to read {event_dir}: {exc}")
            continue

        if tag not in accumulator.Tags().get("scalars", []):
            continue

        for event in accumulator.Scalars(tag):
            points.append((int(event.step), float(event.value), float(event.wall_time)))

    # If a run was resumed, duplicate steps can exist. Keep the latest wall-time value.
    latest_by_step: dict[int, tuple[float, float]] = {}
    for step, value, wall_time in points:
        current = latest_by_step.get(step)
        if current is None or wall_time >= current[1]:
            latest_by_step[step] = (value, wall_time)

    return [(step, value, wall_time) for step, (value, wall_time) in sorted(latest_by_step.items())]


def write_reward_json(
    experiment_dir: Path,
    output_dir: Path,
    tag: str,
    precision: int,
) -> bool:
    points = collect_reward_points(experiment_dir, tag)
    if not points:
        print(f"[SKIP] {experiment_dir.name}: no points for tag {tag}")
        return False

    steps = [step for step, _, _ in points]
    rewards = [round(value, max(precision, 0)) for _, value, _ in points]
    output_path = output_dir / f"{experiment_dir.name}_reward.json"
    output_path.write_text(json.dumps([steps, rewards], separators=(",", ":")), encoding="utf-8")
    print(f"[OK] {experiment_dir.name}: {len(steps)} points -> {output_path}")
    return True


def main() -> None:
    args = parse_args()
    experiment_dirs = find_experiment_dirs(args.log_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for experiment_dir in experiment_dirs:
        if write_reward_json(
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            tag=args.tag,
            precision=args.precision,
        ):
            exported += 1

    print(f"[DONE] Exported {exported}/{len(experiment_dirs)} experiments to {output_dir}")


if __name__ == "__main__":
    main()
