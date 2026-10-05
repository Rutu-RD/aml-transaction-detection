
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd

from aml_detection.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

FEATURES_FILE = "transactions_features.parquet"

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
# All selected by sweeping training data, not chosen by intuition. See
# notebooks/01_eda.ipynb section 7.2 for the full sweeps.
 
#: Minimum prior transactions before a ratio or baseline is meaningful.
#: Below this, 1/1 and 1/2 produce spikes at 1.0 and 0.5 that reflect newness
#: rather than fan-out behaviour.
MIN_PRIOR_TXNS: int = 3
 
#: Rapid pass-through. Lift 3.26x.
#: NOTE: timestamps are minute-granular, so 10s, 30s and 60s return identical
#: results. This means "same or adjacent minute", not literally 60 seconds.
RAPID_TXN_SECONDS: int = 60
 
#: Dormant account reactivating. Lift 10.33x, retaining 20% of all positives.
#: Peak lift is 24.6x at 400,000s but retains only 7% -- coverage preferred.
#: CAVEAT: 200,000s is ~23% of the 10-day observation window. Refit on longer
#: data; the dormancy signal should generalise, the constant will not.
DORMANT_WAKE_SECONDS: int = 200_000
 
#: Fan-in is flat below this and jumps sharply above it (top decile rate
#: 0.0035 versus 0.0004 base). A threshold, not a gradient.
HIGH_FANIN_RATIO: float = 0.667
 
#: Hour 0 carries ~4x the volume of any other hour, which indicates a default
#: timestamp for records without a precise time rather than a midnight surge.
UNKNOWN_HOUR: int = 0
 
#: Formats with zero positives across 597K rows. Excluded from TRAINING rows
#: only -- production must still score them, and zero-Wire-laundering is a
#: simulator artifact (wires are a primary laundering channel in reality).
ZERO_POSITIVE_FORMATS: tuple[str, ...] = ("Reinvestment", "Wire")
 
 
# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
 
def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Account history using ONLY prior transactions.
 
    Each feature answers: what did the bank know about this account
    immediately before this transaction?
 
    PERFORMANCE NOTE
        The obvious implementation of a prior mean,
        `groupby.apply(lambda s: s.shift().expanding().mean())`, runs a Python
        function per group -- minutes across ~500K accounts. Two vectorised
        identities are used instead:
 
        prior sum    = cumsum - current value
        prior count  = cumcount (already excludes current)
 
        Expanding nunique has no pandas equivalent at all. Instead, mark the
        first appearance of each (src, dst) pair and cumsum those markers per
        sender: a repeat pair contributes zero, so the running count only rises
        on genuinely new counterparties. Subtracting the current marker
        excludes the current transaction.
 
        Result: ~5s on 3.5M rows, verified leakage-free against brute-force
        recomputation (see tests/test_features.py).
    """
    logger.info(f"Building account features for {len(df):,} rows")
 
    df = df.sort_values("timestamp").reset_index(drop=True)
    src = df.groupby("src_account", sort=False)
    dst = df.groupby("dst_account", sort=False)
 
    # transaction counts so far (cumcount is 0-based, so already prior-only)
    df["src_txn_count_prior"] = src.cumcount()
    df["dst_txn_count_prior"] = dst.cumcount()
 
    # timing: gap since this account's previous transaction
    df["src_secs_since_last"] = src.timestamp.diff().dt.total_seconds()
    df["dst_secs_since_last"] = dst.timestamp.diff().dt.total_seconds()
    df["src_secs_since_last_log"] = np.log1p(df["src_secs_since_last"])
 
    # distinct counterparties so far
    first_pair = ~df.duplicated(["src_account", "dst_account"])
    running = first_pair.groupby(df["src_account"], sort=False).cumsum()
    df["src_unique_dst_prior"] = running - first_pair.astype(int)
 
    first_pair_in = ~df.duplicated(["dst_account", "src_account"])
    running_in = first_pair_in.groupby(df["dst_account"], sort=False).cumsum()
    df["dst_unique_src_prior"] = running_in - first_pair_in.astype(int)
 
    # fan ratios: counterparty breadth relative to activity
    #   fan-out (scatter) -- one account paying many distinct accounts
    #   fan-in  (gather)  -- many distinct accounts feeding one
    df["src_fanout_ratio"] = np.where(
        df["src_txn_count_prior"] >= MIN_PRIOR_TXNS,
        df["src_unique_dst_prior"] / df["src_txn_count_prior"],
        np.nan,
    )
    df["dst_fanin_ratio"] = np.where(
        df["dst_txn_count_prior"] >= MIN_PRIOR_TXNS,
        df["dst_unique_src_prior"] / df["dst_txn_count_prior"],
        np.nan,
    )
    return df
 
 
def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount signals, normalised so they are comparable across currencies.
 
    Raw amount is meaningless across rows: currency medians span six orders of
    magnitude (Bitcoin 0.07, USD 962, Rupee 69,166, Yen 101,537) and the file
    carries no exchange rates.
 
    amt_pct_ccy (40x spread)
        Percentile rank within currency. Normal transactions are uniform by
        construction, so any departure from flat is the signal. Laundering
        concentrates between 0.6 and 0.85 and then COLLAPSES in the top decile
        -- large enough to be worth moving, small enough to avoid the scrutiny
        the largest transactions attract. Classic structuring, and strongly
        non-monotonic.
 
    src_log_dev (3.5x spread)
        How large is this transaction relative to what this account normally
        sends? Uses a GEOMETRIC mean (average of logs), not an arithmetic one.
 
        This mattered: with an arithmetic mean the distribution centred at -3,
        implying a typical transaction was 5% of the account's average. One
        huge transaction was dragging the mean upward and making everything
        after it look small. Averaging in log space is outlier-resistant --
        for an account sending 100, 100, 100, 1,000,000 the arithmetic mean is
        250,000 while the geometric mean is ~560.
    """
    logger.info("Building amount features")
 
    df["amt_pct_ccy"] = (
        df.groupby("currency_paid", observed=True)["amount_paid"].rank(pct=True)
    )
 
    log_amount = np.log1p(df["amount_paid"])
    grouped = log_amount.groupby(df["src_account"], sort=False)
    prior_log_mean = (
        (grouped.cumsum() - log_amount)
        / df["src_txn_count_prior"].replace(0, np.nan)
    )
    df["src_log_dev"] = np.where(
        df["src_txn_count_prior"] >= MIN_PRIOR_TXNS,
        log_amount - prior_log_mean,
        np.nan,
    )
    return df
 
 
def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day signals.
 
    Cyclical encoding so 23:00 and 00:00 are adjacent rather than 23 units
    apart on a number line.
 
    Only hour-of-day is used. Absolute date and days-since-file-start encode
    position within this particular file and cannot exist in production.
    Day-of-week was rejected: 10 days of data gives 1-2 observations per
    weekday, and the highest-rate days were the lowest-volume days.
    """
    logger.info("Building time features")
 
    hour = df["timestamp"].dt.hour
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_hour_zero"] = hour == UNKNOWN_HOUR
    return df
 
 
def build_threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit boolean flags at the thresholds found in EDA.
 
    Tree models would find these splits themselves, so the flags earn their
    place for two other reasons:
 
    1. Explainability. SHAP on a raw gap gives an investigator
       "src_secs_since_last contributed +0.3". A flag gives them "money moved
       within a minute of arriving". Only one of those justifies opening a
       case, which matters in a regulated workflow where every alert needs a
       written rationale.
 
    2. They encode domain knowledge. With only ~2,850 positives, telling the
       model where to look beats making it discover the boundary unaided.
    """
    logger.info("Building threshold flags")
 
    df["src_rapid_txn"] = df["src_secs_since_last"] <= RAPID_TXN_SECONDS
    df["src_dormant_wake"] = df["src_secs_since_last"] > DORMANT_WAKE_SECONDS
    df["dst_high_fanin"] = df["dst_fanin_ratio"] > HIGH_FANIN_RATIO
    return df
 
 
# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
 
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature group, in dependency order.
 
    Order matters: amount features need src_txn_count_prior, and the threshold
    flags need src_secs_since_last and dst_fanin_ratio. Account features must
    therefore run first.
 
    Must be called on the FULL chronological frame, before splitting.
    """
    logger.info("Building all features")
 
    out = (
        df.pipe(build_account_features)
        .pipe(build_amount_features)
        .pipe(build_time_features)
        .pipe(build_threshold_flags)
    )
 
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Feature build did not produce: {missing}")
 
    logger.success(f"Built {len(FEATURE_COLUMNS)} features on {len(out):,} rows")
    return out
 
 
# --------------------------------------------------------------------------
# The model's inputs
# --------------------------------------------------------------------------
 
#: Measured lift over the 0.089% base rate, from notebooks/01_eda.ipynb.
NUMERIC_FEATURES: tuple[str, ...] = (
    "amt_pct_ccy",              # 40x  -- non-monotonic band (structuring)
    "src_secs_since_last_log",  # 60x  -- U-shaped, strongest single feature
    "dst_fanin_ratio",          # 8.7x -- threshold, gather typology
    "src_fanout_ratio",         # 4.2x -- threshold, scatter typology
    "src_log_dev",              # 3.5x -- marginal, rho=0.76 with amt_pct_ccy
    "hour_sin",                 # 6x   -- cyclical
    "hour_cos",
)
 
BOOLEAN_FEATURES: tuple[str, ...] = (
    "src_rapid_txn",            # 3.3x
    "src_dormant_wake",         # 10.3x
    "dst_high_fanin",
    "is_hour_zero",             # data-quality indicator
)
 
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "payment_format",           # 7.3x -- ACH holds 83% of positives
    "src_entity_type",          # ~1.3x, weak context
    "dst_entity_type",
    "src_bank_country",
    "dst_bank_country",
)
 
FEATURE_COLUMNS: tuple[str, ...] = (
    NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
)
 
TARGET: str = "is_laundering"
 
 
def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only. Raises if the frame has not been through
    build_features."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing feature columns: {missing}. "
            "Run build_features() on the full frame first."
        )
    return df[list(FEATURE_COLUMNS)]


def save_features(df: pd.DataFrame, filename: str = FEATURES_FILE) -> Path:
    """Write the feature frame to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_parquet(path, index=False)
    logger.success(
        f"Wrote {len(df):,} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path

if __name__ == "__main__":
    merged = pd.read_parquet(
        INTERIM_DATA_DIR / "merged" / "transactions_merged.parquet"
    )
    save_features(build_features(merged))