#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"
}
DEFAULT_SPLITS = ("train", "test", "val")


def normalize_count(value: Optional[int]) -> Optional[int]:
    """None or a negative value means selecting all available items."""
    if value is None or value < 0:
        return None
    return value


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def canonical_header(name: str) -> str:
    """Normalize headers such as 'Participant ID' and 'participant_id'."""
    return "".join(character.lower() for character in str(name) if character.isalnum())


def find_column(columns: Sequence[str], aliases: Sequence[str]) -> str:
    normalized = {canonical_header(name): name for name in columns}
    for alias in aliases:
        match = normalized.get(canonical_header(alias))
        if match is not None:
            return match
    raise ValueError(
        f"Could not find any of columns {list(aliases)}. "
        f"Available columns: {list(columns)}"
    )


def resolve_asl_citizen_root(dataset_root: Path) -> Path:
    """Accept either the extracted ASL_Citizen directory or its parent."""
    candidates = (dataset_root, dataset_root / "ASL_Citizen")
    for candidate in candidates:
        if (candidate / "splits").is_dir() and (candidate / "videos").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the ASL Citizen structure. Expected either:\n"
        f"  {dataset_root}/splits and {dataset_root}/videos\n"
        "or:\n"
        f"  {dataset_root}/ASL_Citizen/splits and "
        f"{dataset_root}/ASL_Citizen/videos"
    )


def load_split_dataframe(dataset_root: Path, split: str) -> pd.DataFrame:
    """Load and normalize one official ASL Citizen split CSV with pandas."""
    csv_path = dataset_root / "splits" / f"{split}.csv"
    videos_root = dataset_root / "videos"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {csv_path}")

    dataframe = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.columns.empty:
        raise ValueError(f"CSV has no header: {csv_path}")

    participant_column = find_column(
        dataframe.columns, ("Participant ID", "participant_id", "signer_id")
    )
    video_column = find_column(
        dataframe.columns, ("Video file", "video_file", "video", "filename")
    )
    gloss_column = find_column(
        dataframe.columns, ("Gloss", "gloss", "text", "label")
    )
    asl_lex_column = find_column(
        dataframe.columns, ("ASL-LEX Code", "asl_lex_code", "asllexcode")
    )

    dataframe = dataframe.rename(
        columns={
            participant_column: "participant_id",
            video_column: "video_file",
            gloss_column: "gloss",
            asl_lex_column: "asl_lex_code",
        }
    )[["participant_id", "video_file", "gloss", "asl_lex_code"]].copy()

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    invalid_mask = dataframe["video_file"].eq("") | dataframe["gloss"].eq("")
    if invalid_mask.any():
        csv_rows = (dataframe.index[invalid_mask] + 2).tolist()
        preview = ", ".join(map(str, csv_rows[:10]))
        suffix = "..." if len(csv_rows) > 10 else ""
        raise ValueError(
            f"Missing video filename or gloss in {csv_path}; CSV rows: "
            f"{preview}{suffix}"
        )

    dataframe.insert(0, "split", split)
    dataframe["source_video"] = dataframe["video_file"].map(
        lambda name: videos_root / name
    )
    return dataframe.reset_index(drop=True)


def select_classes(
    dataframes_by_split: dict[str, pd.DataFrame],
    class_cnt: Optional[int],
    random_seed: int,
    require_all_splits: bool,
) -> list[str]:
    """Select glosses from the intersection or union of official splits."""
    class_sets = [set(frame["gloss"].unique()) for frame in dataframes_by_split.values()]
    available_classes = (
        set.intersection(*class_sets) if require_all_splits else set.union(*class_sets)
    )

    if not available_classes:
        raise ValueError("No eligible glosses were found in the split CSV files.")

    classes = pd.Series(sorted(available_classes), dtype="string")
    classes = classes.sample(frac=1, random_state=random_seed).tolist()

    class_cnt = normalize_count(class_cnt)
    if class_cnt is not None:
        if class_cnt > len(classes):
            raise ValueError(
                f"class_cnt={class_cnt}, but only {len(classes)} eligible glosses exist."
            )
        classes = classes[:class_cnt]

    return classes


def select_records(
    dataframe: pd.DataFrame,
    selected_classes: Sequence[str],
    videos_per_class: Optional[int],
    random_seed: int,
    split: str,
    allow_fewer: bool,
) -> pd.DataFrame:
    """Select a reproducible number of rows per gloss using pandas."""
    videos_per_class = normalize_count(videos_per_class)
    selected = dataframe[dataframe["gloss"].isin(selected_classes)].copy()

    counts = selected.groupby("gloss", sort=False).size()
    missing_classes = [name for name in selected_classes if name not in counts.index]
    if missing_classes:
        selected = selected[selected["gloss"].isin(counts.index)]

    if videos_per_class is not None and not allow_fewer:
        insufficient = counts[counts < videos_per_class]
        if not insufficient.empty:
            class_name = str(insufficient.index[0])
            available = int(insufficient.iloc[0])
            raise ValueError(
                f"Split='{split}', gloss='{class_name}': requested "
                f"{videos_per_class} videos, but only {available} exist. "
                "Use --allow-fewer to keep all available videos instead."
            )

    class_order = {class_name: index for index, class_name in enumerate(selected_classes)}
    selected["_class_order"] = selected["gloss"].map(class_order)

    sampled_groups: list[pd.DataFrame] = []
    for class_name, group in selected.groupby("gloss", sort=False):
        sample_size = len(group)
        if videos_per_class is not None:
            sample_size = min(videos_per_class, sample_size)

        class_seed = (
            random_seed
            + sum((index + 1) * ord(character) for index, character in enumerate(class_name))
        ) % (2**32 - 1)
        sampled_groups.append(group.sample(n=sample_size, random_state=class_seed))

    if not sampled_groups:
        return selected.iloc[0:0].drop(columns="_class_order")

    result = pd.concat(sampled_groups, ignore_index=True)
    result = result.sort_values(["_class_order"], kind="stable")
    return result.drop(columns="_class_order").reset_index(drop=True)


def safe_remove_output(dataset_root: Path, output_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()

    if dataset_root == output_root:
        raise ValueError("dataset_root and output_root cannot be the same directory.")
    if output_root == Path(output_root.anchor):
        raise ValueError("The filesystem root cannot be used as output_root.")
    if dataset_root in output_root.parents:
        raise ValueError(
            "output_root cannot be inside dataset_root because deleting/recreating "
            "the output could damage the source dataset."
        )

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def copy_video(source: Path, destination: Path, use_symlinks: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_symlinks:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def process_split(
    output_root: Path,
    split: str,
    dataframe: pd.DataFrame,
    use_symlinks: bool,
    skip_missing: bool,
) -> tuple[pd.DataFrame, int]:
    """Copy or link one DataFrame-defined split into split/gloss/video folders."""
    started_at = time.perf_counter()
    written_indices: list[int] = []
    missing_count = 0

    rows = dataframe.itertuples(index=True, name="ASLCitizenRow")
    progress_bar = tqdm(
        rows,
        total=len(dataframe),
        desc=f"Preparing {split}",
        unit="video",
        dynamic_ncols=True,
    )

    for row in progress_bar:
        source_video = Path(row.source_video)
        if not source_video.is_file():
            if skip_missing:
                missing_count += 1
                continue
            raise FileNotFoundError(
                f"Video listed in {split}.csv was not found: {source_video}"
            )

        if source_video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {source_video}")

        destination = output_root / split / row.gloss / row.video_file
        copy_video(source_video, destination, use_symlinks)
        written_indices.append(row.Index)
        progress_bar.set_postfix(gloss=row.gloss, video=row.video_file)

    written = dataframe.loc[written_indices].copy().reset_index(drop=True)
    elapsed = time.perf_counter() - started_at
    print(
        f"[{split.upper()}] Completed: {len(written)} videos, "
        f"{missing_count} missing, {format_duration(elapsed)}"
    )
    return written, missing_count


def write_manifest(output_root: Path, dataframe: pd.DataFrame) -> None:
    """Write metadata for the prepared subset using pandas."""
    manifest = dataframe[
        ["split", "gloss", "participant_id", "asl_lex_code", "video_file"]
    ].copy()
    manifest["video_path"] = (
        manifest["split"]
        + "/"
        + manifest["gloss"]
        + "/"
        + manifest["video_file"]
    )

    manifest_path = output_root / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False, encoding="utf-8")
    print(f"Manifest created: {manifest_path}")


def print_split_statistics(dataframes_by_split: dict[str, pd.DataFrame]) -> None:
    summary = pd.DataFrame(
        {
            split: {
                "rows": len(frame),
                "glosses": frame["gloss"].nunique(),
                "participants": frame["participant_id"].nunique(),
            }
            for split, frame in dataframes_by_split.items()
        }
    ).T
    # print("\nSource split statistics:")
    # print(summary.to_string())
    # print()


def prepare_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    class_cnt: Optional[int] = None,
    train_cnt: Optional[int] = None,
    test_cnt: Optional[int] = None,
    val_cnt: Optional[int] = None,
    random_seed: int = 42,
    use_symlinks: bool = False,
    require_all_splits: bool = True,
    allow_fewer: bool = False,
    skip_missing: bool = False,
) -> None:
    """Prepare a class-folder subset from the official ASL Citizen release."""
    requested_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset_root = resolve_asl_citizen_root(requested_root)

    dataframes_by_split = {
        split: load_split_dataframe(dataset_root, split) for split in DEFAULT_SPLITS
    }
    print_split_statistics(dataframes_by_split)

    selected_classes = select_classes(
        dataframes_by_split,
        class_cnt=class_cnt,
        random_seed=random_seed,
        require_all_splits=require_all_splits,
    )

    print(f"ASL Citizen root: {dataset_root}")
    print(f"Output dataset: {output_root}")
    print(f"Random seed: {random_seed}")
    print(f"Selected glosses: {len(selected_classes)}")
    print(f"Require gloss in every split: {require_all_splits}")

    safe_remove_output(dataset_root, output_root)

    split_counts = {
        "train": normalize_count(train_cnt),
        "test": normalize_count(test_cnt),
        "val": normalize_count(val_cnt),
    }

    started_at = time.perf_counter()
    written_frames: list[pd.DataFrame] = []
    total_missing = 0

    for split_index, split in enumerate(DEFAULT_SPLITS):
        selected = select_records(
            dataframe=dataframes_by_split[split],
            selected_classes=selected_classes,
            videos_per_class=split_counts[split],
            random_seed=random_seed + split_index,
            split=split,
            allow_fewer=allow_fewer,
        )
        written, missing_count = process_split(
            output_root=output_root,
            split=split,
            dataframe=selected,
            use_symlinks=use_symlinks,
            skip_missing=skip_missing,
        )
        written_frames.append(written)
        total_missing += missing_count

    all_written = pd.concat(written_frames, ignore_index=True)
    write_manifest(output_root, all_written)

    elapsed = time.perf_counter() - started_at
    print()
    print(
        f"Dataset preparation completed: {len(selected_classes)} glosses, "
        f"{len(all_written)} videos, {total_missing} missing."
    )
    print(f"Total time: {format_duration(elapsed)}")


def parse_optional_count(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"all", "none", "-1"}:
        return None
    parsed = int(value)
    return None if parsed < 0 else parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an ASL Citizen subset with pandas while retaining its "
            "official signer-independent train/val/test splits."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-cnt", type=parse_optional_count, default=None)
    parser.add_argument("--train-cnt", type=parse_optional_count, default=None)
    parser.add_argument("--test-cnt", type=parse_optional_count, default=None)
    parser.add_argument("--val-cnt", type=parse_optional_count, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying videos.",
    )
    parser.add_argument(
        "--allow-missing-split-classes",
        action="store_true",
        help=(
            "Select glosses from the union of splits. By default a selected gloss "
            "must occur in train, val, and test."
        ),
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help=(
            "When a gloss has fewer videos than requested in a split, keep all "
            "available videos instead of raising an error."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip CSV entries whose source video file is missing.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one gloss and at most one video per split.",
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()

    if arguments.smoke_test:
        class_cnt = train_cnt = test_cnt = val_cnt = 1
        allow_fewer = True
    else:
        class_cnt = arguments.class_cnt
        train_cnt = arguments.train_cnt
        test_cnt = arguments.test_cnt
        val_cnt = arguments.val_cnt
        allow_fewer = arguments.allow_fewer

    prepare_dataset(
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        class_cnt=class_cnt,
        train_cnt=train_cnt,
        test_cnt=test_cnt,
        val_cnt=val_cnt,
        random_seed=arguments.random_seed,
        use_symlinks=arguments.symlink,
        require_all_splits=not arguments.allow_missing_split_classes,
        allow_fewer=allow_fewer,
        skip_missing=arguments.skip_missing,
    )


if __name__ == "__main__":
    main()