#!/usr/bin/env python3
"""Export eval reward curves from TensorBoard event files.

By default, run this from the workspace root:

    python scripts/export_eval_reward_curve.py

It scans tensorboard_log/easy_r1/* and writes one tiny txt file per
experiment into extract_results/.

Each output file is intentionally tiny for restricted transfers:

    xaxis: [0, 5, 10]
    yaxis: [0.12, 0.34, 0.56]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


DEFAULT_TAG = "val/reward_score"
DEFAULT_LOGDIR = Path("tensorboard_log/easy_r1")
DEFAULT_OUTPUT_DIR = Path("extract_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export eval reward scalar curves as small txt files.")
    parser.add_argument(
        "logdir",
        nargs="?",
        type=Path,
        default=DEFAULT_LOGDIR,
        help=f"TensorBoard log root to scan. Default: {DEFAULT_LOGDIR}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for txt files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"TensorBoard scalar tag to export. Default: {DEFAULT_TAG}",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for y values. Default: 6",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=10_000,
        help="Maximum output size in bytes; points are uniformly downsampled if needed. Default: 10000",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List scalar tags found under logdir and exit.",
    )
    return parser.parse_args()


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard is required to read event files. Try running inside the training env "
            "or install tensorboard."
        ) from exc
    return EventAccumulator


def find_event_dirs(logdir: Path) -> list[Path]:
    if not logdir.exists():
        raise SystemExit(f"Logdir does not exist: {logdir}")
    event_files = sorted(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under: {logdir}")
    return sorted({path.parent for path in event_files})


def find_experiment_dirs(logdir: Path) -> list[Path]:
    """Return experiment directories directly under logdir.

    TensorBoard hparams can create numeric subdirectories below an experiment
    directory. We therefore treat immediate children of logdir as experiments
    instead of every event-file parent.
    """
    if not logdir.exists():
        raise SystemExit(f"Logdir does not exist: {logdir}")
    if list(logdir.glob("events.out.tfevents.*")):
        return [logdir]

    experiments = [
        child
        for child in sorted(logdir.iterdir())
        if child.is_dir() and any(child.rglob("events.out.tfevents.*"))
    ]
    if not experiments:
        raise SystemExit(f"No TensorBoard experiment directories found under: {logdir}")
    return experiments


def collect_scalar_tags(event_dirs: list[Path]) -> dict[str, list[Path]]:
    EventAccumulator = load_event_accumulator()
    tag_to_dirs: dict[str, list[Path]] = {}
    for event_dir in event_dirs:
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:  # pragma: no cover - defensive for partially written event files.
            print(f"[WARN] Failed to read {event_dir}: {exc}")
            continue
        for tag in accumulator.Tags().get("scalars", []):
            tag_to_dirs.setdefault(tag, []).append(event_dir)
    return tag_to_dirs


def collect_points(event_dirs: list[Path], tag: str) -> list[tuple[int, float, float]]:
    EventAccumulator = load_event_accumulator()
    points: list[tuple[int, float, float]] = []
    for event_dir in event_dirs:
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

    # Deduplicate by step. If a run was resumed, keep the latest event for that step.
    latest_by_step: dict[int, tuple[float, float]] = {}
    for step, value, wall_time in points:
        current = latest_by_step.get(step)
        if current is None or wall_time >= current[1]:
            latest_by_step[step] = (value, wall_time)
    return [(step, value, wall_time) for step, (value, wall_time) in sorted(latest_by_step.items())]


def downsample(points: list[tuple[int, float, float]], keep: int) -> list[tuple[int, float, float]]:
    if keep <= 0 or len(points) <= keep:
        return points
    if keep == 1:
        return [points[-1]]
    selected = []
    last_index = len(points) - 1
    for i in range(keep):
        index = round(i * last_index / (keep - 1))
        selected.append(points[index])
    return selected


def render(points: list[tuple[int, float, float]], tag: str, precision: int) -> str:
    xaxis = [step for step, _, _ in points]
    yaxis = [round(value, precision) for _, value, _ in points]
    return f"xaxis: {xaxis}\nyaxis: {yaxis}\n"


def render_with_size_limit(
    points: list[tuple[int, float, float]], tag: str, precision: int, max_bytes: int
) -> tuple[str, int]:
    keep = len(points)
    while True:
        selected = downsample(points, keep)
        text = render(selected, tag=tag, precision=precision)
        if len(text.encode("utf-8")) <= max_bytes or keep <= 2:
            return text, len(selected)
        keep = max(2, math.floor(keep * 0.8))


def export_experiment(
    experiment_dir: Path,
    output_dir: Path,
    tag: str,
    precision: int,
    max_bytes: int,
) -> bool:
    event_dirs = find_event_dirs(experiment_dir)
    points = collect_points(event_dirs, tag)
    if not points:
        print(f"[SKIP] {experiment_dir.name}: no scalar points for tag {tag}")
        return False

    text, kept = render_with_size_limit(
        points,
        tag=tag,
        precision=max(precision, 0),
        max_bytes=max(max_bytes, 100),
    )
    output_path = output_dir / f"{experiment_dir.name}.txt"
    output_path.write_text(text, encoding="utf-8")
    print(
        f"[OK] {experiment_dir.name}: wrote {kept}/{len(points)} points to {output_path} "
        f"({len(text.encode('utf-8'))} bytes)."
    )
    return True


def main() -> None:
    args = parse_args()
    experiment_dirs = find_experiment_dirs(args.logdir)

    if args.list_tags:
        event_dirs = find_event_dirs(args.logdir)
        tag_to_dirs = collect_scalar_tags(event_dirs)
        for tag in sorted(tag_to_dirs):
            print(tag)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for experiment_dir in experiment_dirs:
        if export_experiment(
            experiment_dir=experiment_dir,
            output_dir=args.output_dir,
            tag=args.tag,
            precision=args.precision,
            max_bytes=args.max_bytes,
        ):
            exported += 1

    if exported == 0:
        event_dirs = [event_dir for experiment_dir in experiment_dirs for event_dir in find_event_dirs(experiment_dir)]
        tag_to_dirs = collect_scalar_tags(event_dirs)
        matching_tags = [tag for tag in sorted(tag_to_dirs) if "reward" in tag.lower() or "eval" in tag.lower()]
        hint = "\n".join(f"  {tag}" for tag in matching_tags[:50])
        raise SystemExit(
            f"No experiments contained scalar points for tag: {args.tag}\n"
            f"Use --list-tags to inspect available tags. Reward-like candidates:\n{hint}"
        )

    print(f"[DONE] Exported {exported}/{len(experiment_dirs)} experiments to {args.output_dir}")


if __name__ == "__main__":
    main()
