#!/usr/bin/env python3

"""Generate a large test ZIP file from a set of source files.

This script creates a ZIP file by making multiple copies of the provided input files.
Useful for testing upload handlers with large ZIP archives.

Usage:
uv run scripts/create_big_zip.py test_data/datasets/valid/default.csv \
    test_data/datasets/valid/default.json test_data/datasets/valid/default.xml
-> This creates a ZIP file with 100 copies of each input file.
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def create_test_zip(
    files: list[str],
    copies: int,
    output_path: str = "test_large.zip",
    preserve_temp: bool = False,
) -> Path:
    """Create a ZIP archive with multiple copies of the input files.

    Args:
        files: List of file paths to include
        copies: Number of copies to make of each file
        output_path: Path for the output ZIP file
        preserve_temp: Whether to keep the temp directory after zipping

    Returns:
        Path to the created ZIP file
    """
    temp_dir = tempfile.mkdtemp()
    print(f"Creating temporary directory: {temp_dir}")  # noqa: T201

    try:
        file_count = 0
        for input_file in files:
            if not Path(input_file).exists():
                print(f"Warning: File not found, skipping: {input_file}")  # noqa: T201
                continue

            source_path = Path(input_file)
            base_name = source_path.stem
            extension = source_path.suffix

            for i in range(copies):
                target_name = f"{base_name}_{i}{extension}"
                target_path = Path(temp_dir) / target_name
                shutil.copy2(input_file, target_path)
                file_count += 1

                if file_count % 100 == 0:
                    print(f"Created {file_count} files...")  # noqa: T201

        print(f"Total files created: {file_count}")  # noqa: T201

        output_path = str(Path(output_path))
        print(f"Creating ZIP archive: {output_path}")  # noqa: T201

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, zip_files in os.walk(temp_dir):
                for file in zip_files:
                    file_path = Path(root) / file
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        zip_size = output_path.stat().st_size  # type: ignore
        print(f"ZIP file created: {output_path} ({zip_size / (1024 * 1024):.2f} MB)")  # noqa: T201
        return Path(output_path)

    finally:
        if not preserve_temp:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory removed: {temp_dir}")  # noqa: T201
        else:
            print(f"Temporary directory preserved: {temp_dir}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a large ZIP file by duplicating input files"
    )
    parser.add_argument("files", nargs="+", help="Files to include in the ZIP")
    parser.add_argument(
        "-c",
        "--copies",
        type=int,
        default=100,
        help="Number of copies to make of each file (default: 100)",
    )

    args = parser.parse_args()

    create_test_zip(args.files, args.copies, args.output, args.preserve_temp)
