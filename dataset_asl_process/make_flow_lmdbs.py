#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
}


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def remove_existing_lmdb(lmdb_path: Path) -> None:
    """
    Remove an existing LMDB file/directory and related files.
    """
    related_paths = [
        lmdb_path,
        Path(f"{lmdb_path}-lock"),
        Path(f"{lmdb_path}-order"),
        Path(f"{lmdb_path}-errors.tsv"),
    ]

    for path in related_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def discover_videos(train_root: Path) -> list[Path]:
    """
    Discover videos under:

        train_root/class_name/video.ext
    """
    if not train_root.exists():
        raise FileNotFoundError(
            f"Train directory does not exist: {train_root}"
        )

    videos = sorted(
        path
        for class_directory in train_root.iterdir()
        if class_directory.is_dir()
        for path in class_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.name.startswith(".")
        )
    )

    if not videos:
        raise RuntimeError(
            f"No videos were found under: {train_root}\n"
            "Expected structure: train/class_name/video.mp4"
        )

    return videos


def create_tvl1():
    """
    Create the OpenCV Dual TV-L1 optical-flow estimator.

    Depending on the OpenCV build, the constructor may be exposed
    under cv2.optflow or directly under cv2.
    """
    if (
        hasattr(cv2, "optflow")
        and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create")
    ):
        return cv2.optflow.DualTVL1OpticalFlow_create()

    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        return cv2.DualTVL1OpticalFlow_create()

    raise RuntimeError(
        "Dual TV-L1 optical flow is unavailable. "
        "Install opencv-contrib-python-headless."
    )


def encode_flow_image(
    flow: np.ndarray,
    bound: float,
) -> np.ndarray:
    """
    Convert optical flow to the CoCLR/MemDPC JPEG representation.

    The horizontal and vertical components are clipped to
    [-bound, bound], mapped to [0, 255], and stored in the first
    two OpenCV image channels. The third channel is set to 128.
    """
    flow = np.clip(flow, -bound, bound)
    flow = (flow + bound) * (255.0 / (2.0 * bound))
    flow = np.round(flow).astype(np.uint8)

    height, width = flow.shape[:2]

    flow_image = np.full(
        (height, width, 3),
        fill_value=128,
        dtype=np.uint8,
    )

    flow_image[:, :, 0:2] = flow

    return flow_image


def extract_encoded_flow_frames(
    video_path: Path,
    flow_bound: float,
    jpeg_quality: int,
) -> list[bytes]:
    """
    Compute TV-L1 optical flow for one video and return JPEG bytes.

    A video containing N RGB frames produces approximately N-1
    optical-flow frames.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    success, previous_frame = capture.read()

    if not success or previous_frame is None:
        capture.release()
        raise RuntimeError(
            f"Could not read the first frame: {video_path}"
        )

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY,
    )

    tvl1 = create_tvl1()
    encoded_frames: list[bytes] = []

    encode_parameters = [
        cv2.IMWRITE_JPEG_QUALITY,
        jpeg_quality,
    ]

    try:
        while True:
            success, current_frame = capture.read()

            if not success or current_frame is None:
                break

            current_gray = cv2.cvtColor(
                current_frame,
                cv2.COLOR_BGR2GRAY,
            )

            flow = tvl1.calc(
                previous_gray,
                current_gray,
                None,
            )

            flow_image = encode_flow_image(
                flow=flow,
                bound=flow_bound,
            )

            encoded_success, encoded_image = cv2.imencode(
                ".jpg",
                flow_image,
                encode_parameters,
            )

            if not encoded_success:
                raise RuntimeError(
                    f"Could not encode optical flow: {video_path}"
                )

            encoded_frames.append(encoded_image.tobytes())
            previous_gray = current_gray

    finally:
        capture.release()

    if not encoded_frames:
        raise RuntimeError(
            f"No optical-flow frames were generated: {video_path}"
        )

    return encoded_frames


def create_train_flow_lmdb(
    train_root: str | Path,
    output_lmdb: str | Path,
    flow_bound: float = 20.0,
    jpeg_quality: int = 100,
    map_size_gb: int = 1024,
    commit_interval: int = 10,
    skip_errors: bool = False,
    max_videos: Optional[int] = None,
) -> None:
    """
    Create train_flow.lmdb directly from train videos.

    Parameters
    ----------
    train_root:
        Directory with structure:
            train_root/class_name/video.mp4

    output_lmdb:
        Output LMDB path, for example:
            /data/lmdbs/train_flow.lmdb

    flow_bound:
        Optical-flow clipping bound. CoCLR/MemDPC uses 20.

    jpeg_quality:
        JPEG quality from 1 to 100.

    map_size_gb:
        Maximum virtual LMDB map size in GiB.

    commit_interval:
        Commit the LMDB transaction after this many videos.

    skip_errors:
        Skip unreadable videos instead of terminating.

    max_videos:
        Optional limit for smoke testing.
    """
    train_root = Path(train_root).expanduser().resolve()
    output_lmdb = Path(output_lmdb).expanduser().resolve()

    if flow_bound <= 0:
        raise ValueError("flow_bound must be positive.")

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")

    if map_size_gb <= 0:
        raise ValueError("map_size_gb must be positive.")

    if commit_interval <= 0:
        raise ValueError("commit_interval must be positive.")

    videos = discover_videos(train_root)

    if max_videos is not None:
        if max_videos <= 0:
            raise ValueError("max_videos must be positive.")

        videos = videos[:max_videos]

    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    remove_existing_lmdb(output_lmdb)

    map_size = map_size_gb * 1024**3

    environment = lmdb.open(
        str(output_lmdb),
        subdir=False,
        map_size=map_size,
        readonly=False,
        lock=True,
        meminit=False,
        map_async=True,
    )

    keys: list[bytes] = []
    order: list[str] = []
    video_lengths: list[int] = []
    errors: list[tuple[str, str]] = []

    transaction = environment.begin(write=True)
    started_at = time.perf_counter()

    progress_bar = tqdm(
        videos,
        desc="Creating train_flow.lmdb",
        unit="video",
        dynamic_ncols=True,
    )

    try:
        for video_path in progress_bar:
            class_name = video_path.parent.name
            video_name = video_path.stem
            sample_name = f"{class_name}/{video_name}"

            try:
                flow_frames = extract_encoded_flow_frames(
                    video_path=video_path,
                    flow_bound=flow_bound,
                    jpeg_quality=jpeg_quality,
                )

            except Exception as error:
                errors.append((str(video_path), str(error)))

                if not skip_errors:
                    raise

                tqdm.write(
                    f"[ERROR] Skipping {video_path}: {error}"
                )
                continue

            key = f"{len(keys):09d}".encode("ascii")

            packed_frames = msgpack.dumps(
                flow_frames,
                use_bin_type=True,
            )

            success = transaction.put(
                key,
                packed_frames,
            )

            if not success:
                raise RuntimeError(
                    f"Could not write LMDB entry: {sample_name}"
                )

            keys.append(key)
            order.append(sample_name)
            video_lengths.append(len(flow_frames))

            progress_bar.set_postfix(
                gloss=class_name,
                frames=len(flow_frames),
            )

            if len(keys) % commit_interval == 0:
                transaction.commit()
                transaction = environment.begin(write=True)

        if not keys:
            raise RuntimeError(
                "No videos were successfully written to the LMDB."
            )

        transaction.put(
            b"__keys__",
            msgpack.dumps(keys, use_bin_type=True),
        )
        transaction.put(
            b"__len__",
            msgpack.dumps(len(keys), use_bin_type=True),
        )
        transaction.put(
            b"__order__",
            msgpack.dumps(order, use_bin_type=True),
        )
        transaction.put(
            b"__vlen__",
            msgpack.dumps(video_lengths, use_bin_type=True),
        )

        transaction.commit()
        transaction = None

        environment.sync()

    except Exception:
        if transaction is not None:
            transaction.abort()

        environment.close()
        remove_existing_lmdb(output_lmdb)
        raise

    environment.close()

    Path(f"{output_lmdb}-order").write_text(
        "\n".join(order),
        encoding="utf-8",
    )

    if errors:
        error_path = Path(f"{output_lmdb}-errors.tsv")

        with error_path.open("w", encoding="utf-8") as file:
            file.write("video_path\terror\n")

            for video_path, error_message in errors:
                file.write(
                    f"{video_path}\t{error_message}\n"
                )

        print(f"Error log: {error_path}")

    elapsed = time.perf_counter() - started_at

    print()
    print("train_flow.lmdb creation completed.")
    print(f"Output: {output_lmdb}")
    print(f"Videos written: {len(keys)}")
    print(f"Flow frames written: {sum(video_lengths)}")
    print(f"Failed videos: {len(errors)}")
    print(f"Elapsed time: {format_duration(elapsed)}")


def verify_lmdb(lmdb_path: str | Path) -> None:
    """
    Verify LMDB metadata and decode the first stored JPEG.
    """
    lmdb_path = Path(lmdb_path).expanduser().resolve()

    environment = lmdb.open(
        str(lmdb_path),
        subdir=lmdb_path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    try:
        with environment.begin(write=False) as transaction:
            packed_length = transaction.get(b"__len__")
            packed_keys = transaction.get(b"__keys__")
            packed_order = transaction.get(b"__order__")
            packed_vlen = transaction.get(b"__vlen__")

            if any(
                value is None
                for value in (
                    packed_length,
                    packed_keys,
                    packed_order,
                    packed_vlen,
                )
            ):
                raise RuntimeError(
                    "Required LMDB metadata is missing."
                )

            length = msgpack.loads(
                packed_length,
                raw=False,
            )
            keys = msgpack.loads(
                packed_keys,
                raw=False,
            )
            order = msgpack.loads(
                packed_order,
                raw=False,
            )
            video_lengths = msgpack.loads(
                packed_vlen,
                raw=False,
            )

            if length != len(keys):
                raise RuntimeError(
                    "__len__ does not match __keys__."
                )

            if length != len(order):
                raise RuntimeError(
                    "__len__ does not match __order__."
                )

            if length != len(video_lengths):
                raise RuntimeError(
                    "__len__ does not match __vlen__."
                )

            first_video = msgpack.loads(
                transaction.get(keys[0]),
                raw=False,
            )

            if not first_video:
                raise RuntimeError(
                    "The first LMDB video contains no flow frames."
                )

            first_jpeg = np.frombuffer(
                first_video[0],
                dtype=np.uint8,
            )

            decoded = cv2.imdecode(
                first_jpeg,
                cv2.IMREAD_COLOR,
            )

            if decoded is None:
                raise RuntimeError(
                    "Could not decode the first flow JPEG."
                )

            first_name = order[0]

            if isinstance(first_name, bytes):
                first_name = first_name.decode("utf-8")

            print()
            print("LMDB verification passed.")
            print(f"Videos: {length}")
            print(f"First video: {first_name}")
            print(f"First video flow frames: {len(first_video)}")
            print(f"Flow image shape: {decoded.shape}")

    finally:
        environment.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CoCLR-compatible train_flow.lmdb "
            "directly from train videos."
        )
    )

    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help=(
            "Train dataset directory with structure "
            "train_root/class_name/video.mp4"
        ),
    )
    parser.add_argument(
        "--output-lmdb",
        type=Path,
        required=True,
        help="Output path for train_flow.lmdb.",
    )
    parser.add_argument(
        "--flow-bound",
        type=float,
        default=20.0,
        help="Optical-flow clipping bound. Default: 20.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality from 1 to 100. Default: 100.",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=1024,
        help="LMDB virtual map size in GiB. Default: 1024.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=10,
        help="Commit after this many videos. Default: 10.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip videos that cannot be processed.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process only the first N videos.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only one video and verify the LMDB.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    max_videos = 1 if args.smoke_test else args.max_videos

    create_train_flow_lmdb(
        train_root=args.train_root,
        output_lmdb=args.output_lmdb,
        flow_bound=args.flow_bound,
        jpeg_quality=args.jpeg_quality,
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
        skip_errors=args.skip_errors,
        max_videos=max_videos,
    )

    verify_lmdb(args.output_lmdb)


if __name__ == "__main__":
    main()
