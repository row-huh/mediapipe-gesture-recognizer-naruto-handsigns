from __future__ import annotations

import pathlib
import shutil

import click
import numpy as np
from tqdm import tqdm


def ask_directory(prompt: str, must_exist: bool = True) -> pathlib.Path:
    while True:
        path = pathlib.Path(click.prompt(prompt)).expanduser()

        if must_exist:
            if path.exists() and path.is_dir():
                return path
            click.secho("Directory doesn't exist.\n", fg="red")
        else:
            return path


@click.command()
def main():
    click.echo("\n=== Dataset Splitter ===\n")

    input_root = ask_directory("Dataset root directory")
    output_root = ask_directory("Output directory", must_exist=False)

    num_splits = click.prompt(
        "\nHow many dataset splits?",
        type=click.IntRange(min=1),
    )

    split_sizes = {}

    for i in range(num_splits):
        click.echo(f"\nSplit {i + 1}")
        name = click.prompt("  Name (train/test/val/etc)")
        size = click.prompt(
            "  Images per class",
            type=click.IntRange(min=1),
        )
        split_sizes[name] = size

    seed = click.prompt(
        "\nRandom seed (-1 for random)",
        default=-1,
        show_default=True,
    )
    seed = None if seed == -1 else seed

    click.echo("\nCreating splits...\n")

    rng = np.random.default_rng(seed)

    for class_dir in tqdm(sorted(input_root.iterdir()), desc="Classes"):
        if not class_dir.is_dir():
            continue

        files = sorted(class_dir.iterdir())

        total_requested = sum(split_sizes.values())
        if total_requested > len(files):
            click.secho(
                f"Skipping '{class_dir.name}': "
                f"requested {total_requested} images but only found {len(files)}.",
                fg="yellow",
            )
            continue

        indices = rng.permutation(len(files))

        offset = 0

        for split_name, count in split_sizes.items():
            out_dir = (
                output_root.with_name(f"{output_root.name}_{split_name}")
                / class_dir.name
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx in indices[offset : offset + count]:
                shutil.copy2(files[idx], out_dir / files[idx].name)

            offset += count

    click.secho("\nDone!", fg="green")


if __name__ == "__main__":
    main()