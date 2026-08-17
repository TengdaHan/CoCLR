
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import transforms

import dataset.lmdb_dataset as lmdb_dataset



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an ASL Citizen subset with pandas while retaining its "
            "official signer-independent train/val/test splits."
        )
    )
    parser.add_argument("--lmdb-path", type=Path, default="./data/outputs/lmdbs/train.lmdb", help="Path to the LMDB dataset.")
    return parser


def main():
    arguments = build_argument_parser().parse_args()

    dataset = lmdb_dataset.CustomASLLMDB_2CLIP(
        db_path=arguments.lmdb_path,
        transform=transforms.Compose([
            lambda images: [
                transforms.ToTensor()(image)
                for image in images
            ],
        ]),
        mode="train",
        num_frames=32,
        ds=1,
        return_label=True,
        return_path=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_class}")
    print(f"Classes: {dataset.class_to_idx}")

    sequence, target = dataset[0]
    label, video_name = target

    print(f"Sequence shape: {sequence.shape}")
    print(f"Label: {label}")
    print(f"Class name: {dataset.decode_action(label)}")
    print(f"Video name: {video_name}")

    expected_frames = 32 * 2

    assert isinstance(sequence, torch.Tensor)
    assert sequence.ndim == 4
    assert sequence.shape[1] == expected_frames

    print("Custom LMDB smoke test passed.")


if __name__ == "__main__":
    main()

