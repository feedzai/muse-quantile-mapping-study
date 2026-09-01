#!/usr/bin/env python3
"""Download the BAF dataset suite from Kaggle and report its local path.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME /
KAGGLE_KEY env vars) since the dataset is gated behind a Kaggle login,
even though it is publicly downloadable free of charge.

Usage:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import sys

from muse.data import VARIANTS, download_baf


def main() -> int:
    print(f"Downloading BAF dataset ({', '.join(VARIANTS)}) via kagglehub...")
    try:
        path = download_baf()
    except Exception as exc:  # noqa: BLE001 - surface a clear message to the user
        print(
            "Failed to download BAF automatically. This usually means Kaggle "
            "API credentials are not configured.\n"
            "1. Create an API token at https://www.kaggle.com/settings -> 'Create New Token'.\n"
            "2. Save it to ~/.kaggle/kaggle.json (chmod 600), or export "
            "KAGGLE_USERNAME / KAGGLE_KEY.\n"
            "3. Re-run this script.\n\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"BAF dataset available at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
