from __future__ import annotations

import argparse

from app.index_pipeline import build_index_from_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from documents directory.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Relative path to data directory (default: project data/).",
    )
    args = parser.parse_args()

    chunks = build_index_from_dir(args.data_dir)
    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()

