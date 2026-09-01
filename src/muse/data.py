"""Load the public Bank Account Fraud (BAF) dataset (Jesus et al.,
NeurIPS 2022 Datasets & Benchmarks).

BAF ships as 7 variants (Base, Variant I-V) of ~1M rows each, spanning 8
simulated months (column `month`, values 0-7) with a binary `fraud_bool`
label and ~30 anonymized features. Source: Kaggle
`sgpjesus/bank-account-fraud-dataset-neurips-2022` (CC BY-NC 4.0, see the
upstream repository https://github.com/feedzai/bank-account-fraud for the
full license and datasheet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KAGGLE_HANDLE = "sgpjesus/bank-account-fraud-dataset-neurips-2022"

# Columns we actually rely on; validated at load time so a schema drift in the
# upstream dataset fails loudly instead of silently producing garbage scores.
REQUIRED_COLUMNS = {"fraud_bool", "month"}

VARIANTS = ("Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V")


def _variant_filename(variant: str) -> str:
    return f"{variant}.csv"


def download_baf() -> Path:
    """Download the BAF dataset suite via kagglehub, return the local dir.

    kagglehub manages its own cache location (no supported way to point it
    at a custom directory as of this writing -- see
    https://github.com/Kaggle/kagglehub/issues/214); repeated calls reuse
    that cache automatically.
    """
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_HANDLE)
    return Path(path)


def load_variant(variant: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load one BAF variant as a DataFrame.

    Parameters
    ----------
    variant : one of VARIANTS (e.g. "Base", "Variant II").
    data_dir : directory containing the downloaded CSVs. If None, downloads
        (or reuses the kagglehub cache) automatically.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown BAF variant '{variant}', expected one of {VARIANTS}")

    if data_dir is None:
        data_dir = download_baf()
    data_dir = Path(data_dir)

    path = data_dir / _variant_filename(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected BAF variant file at {path}. Run scripts/download_data.py "
            f"first, or pass an explicit data_dir pointing at the extracted CSVs."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"BAF file {path} is missing expected columns: {missing}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns: everything except label and bookkeeping."""
    exclude = {"fraud_bool", "month"}
    return [c for c in df.columns if c not in exclude]
