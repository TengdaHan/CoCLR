#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import lmdb
import msgpack
import msgpack_numpy
from tqdm import tqdm

msgpack_numpy.patch()


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
}


@dataclass(frozen=True)
class VideoSample:
    split: str
    gloss: str
    video_dir: Path

    @property
    def relative_name(self) -> str:
        """
        CoCLR __order__ format:

            gloss_name/video_directory_name
        """
        return f"{self.gloss}/{self.video_dir.name}"


def normalize_count(value: Optional[int]) -> Optional[int]:
    """
    None yoki manfiy qiymat barcha mavjud elementlarni olishni anglatadi.
    """
    if value is None or value < 0:
        return None

    return value


def remove_existing_lmdb(lmdb_path: Path) -> None:
    """
    Oldingi LMDB va unga tegishli yordamchi fayllarni o'chiradi.
    """
    targets = [
        lmdb_path,
        Path(f"{lmdb_path}-lock"),
        Path(f"{lmdb_path}-order"),
        Path(f"{lmdb_path}-manifest.txt"),
    ]

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def get_gloss_directories(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split papkasi topilmadi: {split_dir}"
        )

    gloss_dirs = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    return gloss_dirs


def get_video_directories(gloss_dir: Path) -> list[Path]:
    """
    Faqat ichida JPEG freym mavjud bo'lgan papkalarni qaytaradi.
    """
    video_dirs: list[Path] = []

    for path in sorted(gloss_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue

        has_frames = any(
            child.is_file()
            and child.suffix.lower() in IMAGE_EXTENSIONS
            for child in path.iterdir()
        )

        if has_frames:
            video_dirs.append(path)

    return video_dirs


def read_video_frames(video_dir: Path) -> list[bytes]:
    """
    Video papkasidagi JPEG freymlarni tartib bilan raw bytes sifatida o'qiydi.
    """
    frame_paths = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not frame_paths:
        raise ValueError(
            f"JPEG freymlar topilmadi: {video_dir}"
        )

    frames: list[bytes] = []

    for frame_path in frame_paths:
        with frame_path.open("rb") as file:
            frames.append(file.read())

    return frames


def choose_classes(
    dataset_root: Path,
    splits: Sequence[str],
    class_cnt: Optional[int],
    random_generator: random.Random,
) -> list[str]:
    """
    Splitlar orasidagi umumiy class/gloss nomlaridan classlarni tanlaydi.

    Bu train, val va test LMDBlarida bir xil classlar bo'lishini ta'minlaydi.
    """
    split_class_sets: list[set[str]] = []

    for split in splits:
        split_dir = dataset_root / split
        gloss_names = {
            directory.name
            for directory in get_gloss_directories(split_dir)
        }
        split_class_sets.append(gloss_names)

    if not split_class_sets:
        raise ValueError("Hech qanday split berilmadi.")

    common_classes = set.intersection(*split_class_sets)

    if not common_classes:
        raise ValueError(
            "Train, val va test splitlari orasida umumiy class topilmadi."
        )

    classes = sorted(common_classes)
    random_generator.shuffle(classes)

    class_cnt = normalize_count(class_cnt)

    if class_cnt is not None:
        if class_cnt > len(classes):
            raise ValueError(
                f"class_cnt={class_cnt}, lekin barcha splitlarda mavjud "
                f"umumiy classlar soni faqat {len(classes)}."
            )

        classes = classes[:class_cnt]

    return classes


def choose_split_samples(
    dataset_root: Path,
    split: str,
    selected_classes: Sequence[str],
    videos_per_class: Optional[int],
    random_generator: random.Random,
) -> list[VideoSample]:
    """
    Har bir class uchun kerakli miqdordagi videoni tanlaydi.
    """
    videos_per_class = normalize_count(videos_per_class)
    samples: list[VideoSample] = []

    for gloss in selected_classes:
        gloss_dir = dataset_root / split / gloss
        video_dirs = get_video_directories(gloss_dir)

        random_generator.shuffle(video_dirs)

        if videos_per_class is not None:
            if videos_per_class > len(video_dirs):
                raise ValueError(
                    f"Split='{split}', class='{gloss}': "
                    f"{videos_per_class} ta video so'ralgan, "
                    f"lekin faqat {len(video_dirs)} ta video mavjud."
                )

            video_dirs = video_dirs[:videos_per_class]

        samples.extend(
            VideoSample(
                split=split,
                gloss=gloss,
                video_dir=video_dir,
            )
            for video_dir in video_dirs
        )

    # LMDB indekslarini tasodifiy tartibda berish.
    random_generator.shuffle(samples)

    return samples


def estimate_map_size(samples: Sequence[VideoSample]) -> int:
    """
    LMDB map_size uchun taxminiy joy hisoblaydi.

    JPEG fayllarning jami hajmiga zaxira qo'shiladi.
    Minimum: 1 GiB.
    """
    total_bytes = 0

    for sample in tqdm(
        samples,
        desc="LMDB hajmini hisoblash",
        unit="video",
        leave=False,
    ):
        for frame_path in sample.video_dir.iterdir():
            if (
                frame_path.is_file()
                and frame_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                total_bytes += frame_path.stat().st_size

    minimum_size = 1 * 1024**3
    estimated_size = int(total_bytes * 1.5) + 256 * 1024**2

    return max(minimum_size, estimated_size)


def write_split_lmdb(
    samples: Sequence[VideoSample],
    lmdb_path: Path,
    split: str,
    commit_interval: int = 100,
    map_size: Optional[int] = None,
) -> None:
    """
    Bitta splitni CoCLR formatidagi LMDB sifatida yozadi.
    """
    if not samples:
        raise ValueError(
            f"'{split}' split uchun yoziladigan video mavjud emas."
        )

    remove_existing_lmdb(lmdb_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    if map_size is None:
        map_size = estimate_map_size(samples)

    started_at = time.perf_counter()

    print()
    print(f"[{split.upper()}] LMDB yaratish boshlandi")
    print(f"Manzil: {lmdb_path}")
    print(f"Videolar soni: {len(samples)}")
    print(f"LMDB map_size: {map_size / 1024**3:.2f} GiB")

    environment = lmdb.open(
        str(lmdb_path),
        subdir=False,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
        lock=True,
    )

    keys: list[bytes] = []
    order: list[str] = []
    video_lengths: list[int] = []

    transaction = environment.begin(write=True)

    try:
        progress_bar = tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"{split} LMDB",
            unit="video",
            dynamic_ncols=True,
        )

        for index, sample in progress_bar:
            frames = read_video_frames(sample.video_dir)

            key = f"{index:09d}".encode("ascii")
            packed_frames = msgpack.dumps(
                frames,
                use_bin_type=True,
            )

            success = transaction.put(key, packed_frames)

            if not success:
                raise RuntimeError(
                    f"LMDB'ga yozib bo'lmadi: {sample.relative_name}"
                )

            keys.append(key)
            order.append(sample.relative_name)
            video_lengths.append(len(frames))

            progress_bar.set_postfix(
                gloss=sample.gloss,
                frames=len(frames),
            )

            if (index + 1) % commit_interval == 0:
                transaction.commit()
                transaction = environment.begin(write=True)

        # CoCLR metadata.
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
        remove_existing_lmdb(lmdb_path)
        raise

    environment.close()

    # CoCLR original skriptidagi tashqi order fayli.
    order_path = Path(f"{lmdb_path}-order")
    order_path.write_text(
        "\n".join(order),
        encoding="utf-8",
    )

    # Tekshirish va debugging uchun qo'shimcha manifest.
    manifest_path = Path(f"{lmdb_path}-manifest.txt")
    with manifest_path.open("w", encoding="utf-8") as file:
        file.write("index\tsplit\tgloss\tvideo\tframe_count\n")

        for index, (sample, frame_count) in enumerate(
            zip(samples, video_lengths)
        ):
            file.write(
                f"{index:09d}\t"
                f"{split}\t"
                f"{sample.gloss}\t"
                f"{sample.video_dir.name}\t"
                f"{frame_count}\n"
            )

    elapsed = time.perf_counter() - started_at

    print(
        f"[{split.upper()}] Tugadi: "
        f"{len(samples)} ta video, "
        f"{sum(video_lengths)} ta freym, "
        f"{format_duration(elapsed)}"
    )


def verify_lmdb(lmdb_path: Path) -> None:
    """
    Yaratilgan LMDB metadata va birinchi videoni tekshiradi.
    """
    environment = lmdb.open(
        str(lmdb_path),
        subdir=False,
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
            packed_video_lengths = transaction.get(b"__vlen__")

            if any(
                value is None
                for value in (
                    packed_length,
                    packed_keys,
                    packed_order,
                    packed_video_lengths,
                )
            ):
                raise RuntimeError(
                    f"LMDB metadata to'liq emas: {lmdb_path}"
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
                packed_video_lengths,
                raw=False,
            )

            if length != len(keys):
                raise RuntimeError(
                    f"__len__={length}, lekin keys={len(keys)}"
                )

            if len(order) != length:
                raise RuntimeError(
                    f"Order soni noto'g'ri: {len(order)} != {length}"
                )

            if len(video_lengths) != length:
                raise RuntimeError(
                    "Video uzunliklari soni noto'g'ri."
                )

            if length > 0:
                first_frames_raw = transaction.get(keys[0])

                if first_frames_raw is None:
                    raise RuntimeError(
                        "Birinchi video LMDB'da topilmadi."
                    )

                first_frames = msgpack.loads(
                    first_frames_raw,
                    raw=False,
                )

                if len(first_frames) != video_lengths[0]:
                    raise RuntimeError(
                        "Birinchi videoning frame soni metadata bilan mos emas."
                    )

    finally:
        environment.close()

    print(f"Tekshirildi: {lmdb_path}")


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} soat {minutes} daqiqa {seconds} soniya"

    if minutes:
        return f"{minutes} daqiqa {seconds} soniya"

    return f"{seconds} soniya"


def convert_dataset_to_lmdb(
    dataset_root: str | Path,
    output_root: str | Path,
    class_cnt: Optional[int] = None,
    train_cnt: Optional[int] = None,
    test_cnt: Optional[int] = None,
    val_cnt: Optional[int] = None,
    random_seed: int = 42,
    commit_interval: int = 100,
    verify: bool = True,
) -> None:
    """
    Datasetning train/test/val splitlarini alohida LMDBlarga o'tkazadi.

    Parametrlar:
        dataset_root:
            train/, test/, val/ papkalari joylashgan manzil.

        output_root:
            LMDB fayllari yoziladigan papka.

        class_cnt:
            Nechta umumiy class tanlanishi.
            None yoki -1 = barcha classlar.

        train_cnt, test_cnt, val_cnt:
            Har bir tanlangan classdan tegishli split uchun
            nechta video olinishi.
            None yoki -1 = barcha videolar.

        random_seed:
            Class va videolarni reproduktiv random tanlash uchun seed.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset manzili topilmadi: {dataset_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    split_counts = {
        "train": normalize_count(train_cnt),
        "test": normalize_count(test_cnt),
        "val": normalize_count(val_cnt),
    }

    main_random_generator = random.Random(random_seed)

    selected_classes = choose_classes(
        dataset_root=dataset_root,
        splits=("train", "test", "val"),
        class_cnt=class_cnt,
        random_generator=main_random_generator,
    )

    print(f"Random seed: {random_seed}")
    print(f"Tanlangan classlar soni: {len(selected_classes)}")
    # print("Tanlangan classlar:")
    # for class_name in selected_classes:
    #     print(f"  - {class_name}")

    total_started_at = time.perf_counter()

    for split_index, (split, count) in enumerate(split_counts.items()):
        # Har bir split uchun deterministik, lekin alohida random oqim.
        split_random_generator = random.Random(
            random_seed + split_index + 1
        )

        samples = choose_split_samples(
            dataset_root=dataset_root,
            split=split,
            selected_classes=selected_classes,
            videos_per_class=count,
            random_generator=split_random_generator,
        )

        lmdb_path = output_root / f"{split}.lmdb"

        write_split_lmdb(
            samples=samples,
            lmdb_path=lmdb_path,
            split=split,
            commit_interval=commit_interval,
        )

        if verify:
            verify_lmdb(lmdb_path)

    total_elapsed = time.perf_counter() - total_started_at

    print()
    print(
        "Barcha splitlar tayyorlandi. "
        f"Umumiy vaqt: {format_duration(total_elapsed)}"
    )


def smoke_test(
    dataset_root: str | Path,
    output_root: str | Path,
) -> None:
    """
    Har bir splitdan bitta class va bitta video bilan tezkor test.
    """
    print("Smoke test boshlandi.")

    convert_dataset_to_lmdb(
        dataset_root=dataset_root,
        output_root=output_root,
        class_cnt=1,
        train_cnt=1,
        test_cnt=1,
        val_cnt=1,
        random_seed=42,
        commit_interval=1,
        verify=True,
    )


def parse_optional_count(value: str) -> Optional[int]:
    """
    CLI'da 'all', 'none' yoki -1 qiymatlarini None ga o'tkazadi.
    """
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
            "Frame papkalaridan CoCLR formatidagi "
            "train/test/val LMDB yaratish."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="train/, test/ va val/ joylashgan dataset papkasi.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="LMDB fayllari yoziladigan papka.",
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
        help="Har bir classdan train videolar soni. Default: all.",
    )
    parser.add_argument(
        "--test-cnt",
        type=parse_optional_count,
        default=None,
        help="Har bir classdan test videolar soni. Default: all.",
    )
    parser.add_argument(
        "--val-cnt",
        type=parse_optional_count,
        default=None,
        help="Har bir classdan val videolar soni. Default: all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=100,
        help="Har nechta videodan keyin LMDB commit qilinishi.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 class va har bir splitdan 1 video bilan test.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Yaratilgan LMDB'ni yakuniy tekshirishni o'chiradi.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.commit_interval <= 0:
        parser.error("--commit-interval musbat son bo'lishi kerak.")

    if arguments.smoke_test:
        smoke_test(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
        )
        return

    convert_dataset_to_lmdb(
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        class_cnt=arguments.class_cnt,
        train_cnt=arguments.train_cnt,
        test_cnt=arguments.test_cnt,
        val_cnt=arguments.val_cnt,
        random_seed=arguments.random_seed,
        commit_interval=arguments.commit_interval,
        verify=not arguments.no_verify,
    )


if __name__ == "__main__":
    main()