"""Temporal splitting and training-row selection.
 
WHY TEMPORAL AND NOT RANDOM
---------------------------
Laundering in this data is multi-step chains. A random split assigns earlier
legs of a chain to train and later legs to test, so the model sees part of the
pattern it is being asked to detect. Metrics would be optimistic and the
failure would be invisible.
 
A time-based split mirrors deployment: the model is fitted on the past and
scored on transactions that arrive afterwards.
 
WHY THREE WAY
-------------
Hyperparameter search, threshold selection, and feature ablation all involve
looking at results and changing something. Doing that against the test set
means the reported number is no longer an estimate of unseen performance --
the test set has been fitted to, indirectly, through a human.
 
    train  0-60%   fit model parameters
    val   60-80%   tune hyperparameters, pick the alert threshold, ablate
    test  80-100%  touched ONCE, at the end
 
EXCLUSIONS APPLY TO TRAINING ROWS ONLY
--------------------------------------
Reinvestment, Wire, self-loops and the burn-in period are excluded from train
and validation because they teach the model nothing (zero positives) or come
from a different base rate.
 
They are NOT excluded from test. Production scores every transaction that
arrives, so a test set with the difficult-but-uninformative rows removed would
describe an easier problem than the real one and overstate performance.
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import pandas as pd
from loguru import logger
 
from aml_detection.config import PROCESSED_DATA_DIR
from aml_detection.features import ZERO_POSITIVE_FORMATS
 
# --------------------------------------------------------------------------
# Split boundaries
# --------------------------------------------------------------------------
 
TRAIN_END_QUANTILE: float = 0.60
VAL_END_QUANTILE: float = 0.80
 
SPLIT_DIR = PROCESSED_DATA_DIR / "splits"
 
 
class SplitError(Exception):
    """Raised when a split is empty, misordered, or contains no positives."""
 
 
# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
 
def temporal_boundaries(
    df: pd.DataFrame,
    train_end: float = TRAIN_END_QUANTILE,
    val_end: float = VAL_END_QUANTILE,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Timestamp cut points at the given row quantiles.
 
    Quantiles of the timestamp column rather than fixed dates, because daily
    volume in this file varies more than fivefold (207K to 1.1M rows/day).
    Splitting on calendar days would produce wildly uneven partitions.
    """
    return (
        df["timestamp"].quantile(train_end),
        df["timestamp"].quantile(val_end),
    )
 
 
def split_temporal(
    df: pd.DataFrame,
    train_end: float = TRAIN_END_QUANTILE,
    val_end: float = VAL_END_QUANTILE,
) -> dict[str, pd.DataFrame]:
    """Partition chronologically into train, validation and test.
 
    Must be called on a frame that has already been through
    features.build_features(). Account-history features look backwards, so
    building them per-split would leave validation and test rows with empty
    history.
    """
    train_cut, val_cut = temporal_boundaries(df, train_end, val_end)
    logger.info(f"Split boundaries: train <= {train_cut} < val <= {val_cut} < test")
 
    splits = {
        "train": df[df["timestamp"] <= train_cut],
        "val": df[(df["timestamp"] > train_cut) & (df["timestamp"] <= val_cut)],
        "test": df[df["timestamp"] > val_cut],
    }
    return {name: part.copy() for name, part in splits.items()}
 
 
# --------------------------------------------------------------------------
# Training-row selection
# --------------------------------------------------------------------------
 
def select_training_rows(df: pd.DataFrame,drop_burn_in
                         :bool = True) -> pd.DataFrame:
    """Drop rows that cannot contribute to learning the target.
 
    Reinvestment and Wire
        Zero positives across 597K rows. A format with no positive examples
        teaches the model only that the format is safe, which is a simulator
        artifact -- wires are a primary laundering channel in reality.
 
    Self-loops
        Zero positives, 11.6% of rows. An account paying itself is not a
        transfer between parties, so there is no counterparty to hide behind.
 
    Burn-in period
        Laundering rate 0.039% versus 0.118% afterwards. Chains take days to
        unfold, so the opening days are structurally short of laundering
        rather than genuinely cleaner. Training across both mixes two base
        rates and miscalibrates the model.
 
    

    drop_burn_in is a flag rather than a fixed rule because it is the only
    exclusion with a real cost: it removes 730 of 734 lost positives, i.e. 32%
    of the training positives. The other two filters cost 5 and 0.

    The justification for dropping is a base-rate inconsistency (0.039% during
    burn-in versus 0.118% after), which affects calibration. Since the headline
    metrics here are rank-based (PR-AUC, recall@k), that may not matter. Tested
    empirically rather than assumed -- see the ablation in modeling/train.py.
    """
    required = ("payment_format", "is_self_loop", "is_burn_in", "is_laundering")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SplitError(
            f"Cannot select training rows, missing columns: {missing}. "
            "These are produced by dataset_loader.clean() -- check that the "
            "frame came through the full pipeline."
        )
 
    before = len(df)
    positives_before = int(df["is_laundering"].sum())
 
    keep = (
        ~df["payment_format"].isin(ZERO_POSITIVE_FORMATS)
        & ~df["is_self_loop"]
        & ~df["is_burn_in"]
    )

    if drop_burn_in:
        keep &= ~df["is_burn_in"]
    
    out = df[keep].copy()
 
    logger.info(
        f"Training rows: {len(out):,} of {before:,} kept "
        f"({len(out) / before:.1%}) | "
        f"positives {int(out['is_laundering'].sum()):,} of {positives_before:,} | "
        f"rate {df['is_laundering'].mean():.4%} -> {out['is_laundering'].mean():.4%}"
    )
    return out
 
 
# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
 
def validate_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check the partitions are usable, and fail loudly if not.
 
    Three failure modes worth catching before training rather than after:
      - an empty split, usually a bad quantile
      - overlapping time ranges, which would mean leakage between partitions
      - a split with no positives, which makes PR-AUC undefined
    """
    rows = {}
    for name, part in splits.items():
        if part.empty:
            raise SplitError(f"'{name}' split is empty -- check the quantiles")
 
        positives = int(part["is_laundering"].sum())
        if positives == 0:
            raise SplitError(
                f"'{name}' split contains no positives. PR-AUC is undefined "
                "and any recall metric is meaningless."
            )
        if positives < 100:
            logger.warning(
                f"'{name}' has only {positives} positives -- metrics will be "
                "high variance"
            )
 
        rows[name] = {
            "rows": len(part),
            "positives": positives,
            "rate": part["is_laundering"].mean(),
            "start": part["timestamp"].min(),
            "end": part["timestamp"].max(),
        }
 
    summary = pd.DataFrame(rows).T
 
    ordered = ["train", "val", "test"]
    present = [name for name in ordered if name in splits]
    for earlier, later in zip(present, present[1:]):
        if splits[earlier]["timestamp"].max() > splits[later]["timestamp"].min():
            raise SplitError(
                f"'{earlier}' overlaps '{later}' in time -- the split leaks"
            )
 
    return summary
 
 
# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
 
def save_splits(splits: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write each partition to data/processed/splits/."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
 
    paths = {}
    for name, part in splits.items():
        path = SPLIT_DIR / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths[name] = path
        logger.success(
            f"Wrote {name}: {len(part):,} rows -> {path} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )
    return paths
 
 
def load_split(name: str) -> pd.DataFrame:
    """Read one partition back.
 
    Reading 'test' should happen exactly once, in final evaluation.
    """
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise SplitError(
            f"{path} not found. Run: python -m aml_detection.splitting"
        )
    if name == "test":
        logger.warning(
            "Loading the TEST split. This should happen only in final "
            "evaluation -- use 'val' for tuning and threshold selection."
        )
    return pd.read_parquet(path)
 
 
# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------
 
FEATURES_FILE = PROCESSED_DATA_DIR / "transactions_features.parquet"
 
 
def build_splits(df: pd.DataFrame | None = None, save: bool = True, drop_burn_in: bool =True) -> dict[str, pd.DataFrame]:
    """Split chronologically, then filter train and validation rows."""
    if df is None:
        if not FEATURES_FILE.exists():
            raise SplitError(
                f"{FEATURES_FILE} not found. "
                "Run: python -m aml_detection.features"
            )
        df = pd.read_parquet(FEATURES_FILE)
        logger.info(f"Loaded {len(df):,} rows with features")
 
    splits = split_temporal(df)
 
    # Exclusions on train and val only. Test keeps every row, because
    # production scores everything that arrives.
    for name in ("train", "val"):
        splits[name] = select_training_rows(splits[name],drop_burn_in=drop_burn_in)
 
    summary = validate_splits(splits)
    logger.info("\n" + summary.to_string())
 
    if save:
        save_splits(splits)
    return splits
 
 
if __name__ == "__main__":
    build_splits()