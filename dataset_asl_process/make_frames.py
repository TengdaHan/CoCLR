#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
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


@dataclass(frozen=True)
class VideoSample:
    split: str
    gloss: str
    video_path: Path

    @property
    def output_directory_name(self) -> str:
        return self.video_path.stem


def normalize_count(value: Optional[int]) -> Optional[int]:
    """
    None yoki manfiy son barcha elementlarni olishni anglatadi.
    """
    if value is None or value < 0:
        return None

    return value


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} soat {minutes} daqiqa {seconds} soniya"

    if minutes:
        return f"{minutes} daqiqa {seconds} soniya"

    return f"{seconds} soniya"


def get_class_directories(split_directory: Path) -> list[Path]:
    if not split_directory.exists():
        raise FileNotFoundError(
            f"Split papkasi topilmadi: {split_directory}"
        )

    return sorted(
        path
        for path in split_directory.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def get_video_files(class_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in class_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.name.startswith(".")
        )
    )


def choose_classes(
    dataset_root: Path,
    splits: Sequence[str],
    class_cnt: Optional[int],
    random_generator: random.Random,
) -> list[str]:
    """
    Barcha splitlarda mavjud bo'lgan umumiy classlarni tanlaydi.
    """
    class_sets: list[set[str]] = []

    for split in splits:
        split_directory = dataset_root / split

        class_names = {
            directory.name
            for directory in get_class_directories(split_directory)
        }

        class_sets.append(class_names)

    if not class_sets:
        raise ValueError("Splitlar berilmagan.")

    common_classes = set.intersection(*class_sets)

    if not common_classes:
        raise ValueError(
            "Train, val va test orasida umumiy class topilmadi."
        )

    classes = sorted(common_classes)
    random_generator.shuffle(classes)

    class_cnt = normalize_count(class_cnt)

    if class_cnt is not None:
        if class_cnt > len(classes):
            raise ValueError(
                f"class_cnt={class_cnt}, lekin umumiy classlar soni "
                f"{len(classes)}."
            )

        classes = classes[:class_cnt]

    return classes


def choose_split_videos(
    dataset_root: Path,
    split: str,
    selected_classes: Sequence[str],
    videos_per_class: Optional[int],
    random_generator: random.Random,
) -> list[VideoSample]:
    videos_per_class = normalize_count(videos_per_class)
    samples: list[VideoSample] = []

    for gloss in selected_classes:
        class_directory = dataset_root / split / gloss
        video_files = get_video_files(class_directory)

        random_generator.shuffle(video_files)

        if videos_per_class is not None:
            if videos_per_class > len(video_files):
                raise ValueError(
                    f"Split='{split}', class='{gloss}': "
                    f"{videos_per_class} ta video so'ralgan, "
                    f"lekin {len(video_files)} ta mavjud."
                )

            video_files = video_files[:videos_per_class]

        samples.extend(
            VideoSample(
                split=split,
                gloss=gloss,
                video_path=video_path,
            )
            for video_path in video_files
        )

    random_generator.shuffle(samples)

    return samples


def remove_existing_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def extract_video_frames(
    video_path: Path,
    output_directory: Path,
    frame_step: int = 1,
    target_fps: Optional[float] = None,
    jpeg_quality: int = 95,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
) -> int:
    """
    Bitta videoni JPEG freymlarga ajratadi.

    frame_step:
        Har nechanchi freym saqlanishi.
        Masalan, 1 = barcha freymlar, 2 = har ikkinchi freym.

    target_fps:
        Belgilansa, video shu FPS bo'yicha sample qilinadi.
        target_fps va frame_step birgalikda ishlatilmaydi.
    """
    remove_existing_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        remove_existing_directory(output_directory)
        raise RuntimeError(f"Videoni ochib bo'lmadi: {video_path}")

    original_fps = capture.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 25.0

    if target_fps is not None:
        if target_fps <= 0:
            capture.release()
            remove_existing_directory(output_directory)
            raise ValueError("target_fps musbat bo'lishi kerak.")

        if target_fps > original_fps:
            effective_step = 1
        else:
            effective_step = max(1, round(original_fps / target_fps))
    else:
        effective_step = frame_step

    source_frame_index = 0
    saved_frame_index = 1

    encode_parameters = [
        cv2.IMWRITE_JPEG_QUALITY,
        jpeg_quality,
    ]

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if source_frame_index % effective_step != 0:
                source_frame_index += 1
                continue

            if resize_width is not None or resize_height is not None:
                frame = resize_frame(
                    frame=frame,
                    target_width=resize_width,
                    target_height=resize_height,
                )

            frame_path = (
                output_directory / f"{saved_frame_index:06d}.jpg"
            )

            write_success = cv2.imwrite(
                str(frame_path),
                frame,
                encode_parameters,
            )

            if not write_success:
                raise RuntimeError(
                    f"Freymni yozib bo'lmadi: {frame_path}"
                )

            saved_frame_index += 1
            source_frame_index += 1

    except Exception:
        capture.release()
        remove_existing_directory(output_directory)
        raise

    capture.release()

    saved_count = saved_frame_index - 1

    if saved_count == 0:
        remove_existing_directory(output_directory)
        raise RuntimeError(
            f"Videodan freym olinmadi: {video_path}"
        )

    return saved_count


def resize_frame(
    frame,
    target_width: Optional[int],
    target_height: Optional[int],
):
    """
    Width yoki heightdan faqat bittasi berilsa aspect ratio saqlanadi.
    Ikkalasi berilsa aniq o'lchamga resize qilinadi.
    """
    original_height, original_width = frame.shape[:2]

    if target_width is not None and target_height is not None:
        new_width = target_width
        new_height = target_height

    elif target_width is not None:
        scale = target_width / original_width
        new_width = target_width
        new_height = max(1, round(original_height * scale))

    elif target_height is not None:
        scale = target_height / original_height
        new_width = max(1, round(original_width * scale))
        new_height = target_height

    else:
        return frame

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def process_split(
    samples: Sequence[VideoSample],
    output_root: Path,
    split: str,
    frame_step: int,
    target_fps: Optional[float],
    jpeg_quality: int,
    resize_width: Optional[int],
    resize_height: Optional[int],
    skip_errors: bool,
) -> tuple[int, int]:
    split_started_at = time.perf_counter()

    processed_videos = 0
    total_frames = 0
    failed_videos: list[tuple[Path, str]] = []

    progress_bar = tqdm(
        samples,
        desc=f"{split} freymlar",
        unit="video",
        dynamic_ncols=True,
    )

    for sample in progress_bar:
        output_directory = (
            output_root
            / split
            / sample.gloss
            / sample.output_directory_name
        )

        try:
            frame_count = extract_video_frames(
                video_path=sample.video_path,
                output_directory=output_directory,
                frame_step=frame_step,
                target_fps=target_fps,
                jpeg_quality=jpeg_quality,
                resize_width=resize_width,
                resize_height=resize_height,
            )

            processed_videos += 1
            total_frames += frame_count

            progress_bar.set_postfix(
                gloss=sample.gloss,
                frames=frame_count,
            )

        except Exception as error:
            failed_videos.append(
                (sample.video_path, str(error))
            )

            if not skip_errors:
                raise

            tqdm.write(
                f"[XATO] {sample.video_path}: {error}"
            )

    elapsed = time.perf_counter() - split_started_at

    print(
        f"[{split.upper()}] Tugadi: "
        f"{processed_videos} ta video, "
        f"{total_frames} ta JPEG freym, "
        f"{len(failed_videos)} ta xato, "
        f"{format_duration(elapsed)}"
    )

    if failed_videos:
        error_log_path = output_root / f"{split}_errors.txt"

        with error_log_path.open("w", encoding="utf-8") as file:
            for video_path, error_message in failed_videos:
                file.write(
                    f"{video_path}\t{error_message}\n"
                )

        print(f"Xatolar logi: {error_log_path}")

    return processed_videos, total_frames


def convert_videos_to_frames(
    dataset_root: str | Path,
    output_root: str | Path,
    class_cnt: Optional[int] = None,
    train_cnt: Optional[int] = None,
    test_cnt: Optional[int] = None,
    val_cnt: Optional[int] = None,
    random_seed: int = 42,
    frame_step: int = 1,
    target_fps: Optional[float] = None,
    jpeg_quality: int = 95,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    skip_errors: bool = False,
) -> None:
    """
    Train, test va val videolarni JPEG freymlarga ajratadi.

    Eski output_root to'liq o'chiriladi va qaytadan yaratiladi.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset topilmadi: {dataset_root}"
        )

    if dataset_root == output_root:
        raise ValueError(
            "dataset_root va output_root bir xil bo'lishi mumkin emas."
        )

    if frame_step <= 0:
        raise ValueError("frame_step musbat bo'lishi kerak.")

    if not 1 <= jpeg_quality <= 100:
        raise ValueError(
            "jpeg_quality 1 va 100 orasida bo'lishi kerak."
        )

    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width musbat bo'lishi kerak.")

    if resize_height is not None and resize_height <= 0:
        raise ValueError("resize_height musbat bo'lishi kerak.")

    if target_fps is not None and target_fps <= 0:
        raise ValueError("target_fps musbat bo'lishi kerak.")

    print(f"Eski output o'chirilmoqda: {output_root}")
    remove_existing_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = ("train", "test", "val")
    split_counts = {
        "train": normalize_count(train_cnt),
        "test": normalize_count(test_cnt),
        "val": normalize_count(val_cnt),
    }

    class_random_generator = random.Random(random_seed)

    selected_classes = choose_classes(
        dataset_root=dataset_root,
        splits=splits,
        class_cnt=class_cnt,
        random_generator=class_random_generator,
    )

    print(f"Random seed: {random_seed}")
    print(f"Tanlangan classlar: {len(selected_classes)}")

    # for class_name in selected_classes:
    #     print(f"  - {class_name}")

    total_started_at = time.perf_counter()
    total_videos = 0
    total_frames = 0

    for split_index, split in enumerate(splits):
        split_random_generator = random.Random(
            random_seed + split_index + 1
        )

        samples = choose_split_videos(
            dataset_root=dataset_root,
            split=split,
            selected_classes=selected_classes,
            videos_per_class=split_counts[split],
            random_generator=split_random_generator,
        )

        processed_videos, processed_frames = process_split(
            samples=samples,
            output_root=output_root,
            split=split,
            frame_step=frame_step,
            target_fps=target_fps,
            jpeg_quality=jpeg_quality,
            resize_width=resize_width,
            resize_height=resize_height,
            skip_errors=skip_errors,
        )

        total_videos += processed_videos
        total_frames += processed_frames

    total_elapsed = time.perf_counter() - total_started_at

    print()
    print(
        f"Barcha splitlar tayyor: "
        f"{total_videos} ta video, "
        f"{total_frames} ta JPEG freym."
    )
    print(f"Umumiy vaqt: {format_duration(total_elapsed)}")


def smoke_test(
    dataset_root: str | Path,
    output_root: str | Path,
) -> None:
    """
    1 ta class va har bir splitdan 1 ta video bilan test.
    """
    print("Smoke test boshlandi.")

    convert_videos_to_frames(
        dataset_root=dataset_root,
        output_root=output_root,
        class_cnt=1,
        train_cnt=1,
        test_cnt=1,
        val_cnt=1,
        random_seed=42,
        frame_step=1,
        jpeg_quality=95,
        skip_errors=False,
    )


def parse_optional_count(value: str) -> Optional[int]:
    normalized = value.strip().lower()

    if normalized in {"all", "none", "-1"}:
        return None

    parsed_value = int(value)

    if parsed_value < 0:
        return None

    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Video datasetni CoCLR uchun JPEG freym "
            "papkalariga ajratish."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Asl train/test/val video dataset manzili.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="JPEG freymlar yoziladigan manzil.",
    )
    parser.add_argument(
        "--class-cnt",
        type=parse_optional_count,
        default=None,
        help="Classlar soni. Default: all.",
    )
    parser.add_argument(
        "--train-cnt",
        type=parse_optional_count,
        default=None,
        help="Har bir classdan train video soni. Default: all.",
    )
    parser.add_argument(
        "--test-cnt",
        type=parse_optional_count,
        default=None,
        help="Har bir classdan test video soni. Default: all.",
    )
    parser.add_argument(
        "--val-cnt",
        type=parse_optional_count,
        default=None,
        help="Har bir classdan val video soni. Default: all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "Har nechanchi freym saqlanishi. "
            "1 = barcha freymlar."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Videodan olinadigan taxminiy FPS. "
            "Berilmasa frame-step ishlatiladi."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG sifati: 1-100. Default: 95.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Freymlarning yangi eni.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Freymlarning yangi bo'yi.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Xato videolarni tashlab davom etish.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 class va har bir splitdan 1 video bilan test.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.smoke_test:
        smoke_test(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
        )
        return

    convert_videos_to_frames(
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        class_cnt=arguments.class_cnt,
        train_cnt=arguments.train_cnt,
        test_cnt=arguments.test_cnt,
        val_cnt=arguments.val_cnt,
        random_seed=arguments.random_seed,
        frame_step=arguments.frame_step,
        target_fps=arguments.target_fps,
        jpeg_quality=arguments.jpeg_quality,
        resize_width=arguments.resize_width,
        resize_height=arguments.resize_height,
        skip_errors=arguments.skip_errors,
    )


if __name__ == "__main__":
    main()