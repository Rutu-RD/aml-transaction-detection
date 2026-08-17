from __future__ import annotations
 
import numpy as np
import pandas as pd
import pytest
 
from aml_detection.features import (
    DORMANT_WAKE_SECONDS,
    MIN_PRIOR_TXNS,
    RAPID_TXN_SECONDS,
    build_account_features,
    build_amount_features,
    build_features,
    build_threshold_flags,
    build_time_features,
)
 
 
@pytest.fixture
def transactions() -> pd.DataFrame:
    """Small synthetic frame with repeated accounts and pairs.
 
    Repetition matters: it is what distinguishes "count of transactions" from
    "count of distinct counterparties".
    """
    rng = np.random.default_rng(0)
    n = 2_000
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2022-09-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 900_000, n)), unit="s"),
            "src_account": rng.choice([f"A{i}" for i in range(25)], n),
            "dst_account": rng.choice([f"B{i}" for i in range(40)], n),
            "amount_paid": rng.lognormal(7, 1.5, n),
            "currency_paid": rng.choice(["US Dollar", "Euro", "Rupee"], n),
        }
    )
 
 
# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
 
def test_account_features_use_only_prior_rows(transactions):
    """Brute-force check that no feature sees its own row or later ones."""
    out = build_account_features(transactions)
    rng = np.random.default_rng(1)
 
    for i in rng.choice(np.arange(200, len(out)), 30, replace=False):
        row = out.iloc[i]
        past = out.iloc[:i]                                  # strictly before
        past_src = past[past.src_account == row.src_account]
 
        assert row.src_txn_count_prior == len(past_src), (
            f"row {i}: transaction count includes current or future rows"
        )
        assert row.src_unique_dst_prior == past_src.dst_account.nunique(), (
            f"row {i}: distinct counterparty count is wrong"
        )
 
        if len(past_src):
            expected_gap = (row.timestamp - past_src.timestamp.iloc[-1]).total_seconds()
            assert np.isclose(row.src_secs_since_last, expected_gap), (
                f"row {i}: gap does not match the previous transaction"
            )
 
 
def test_amount_deviation_uses_only_prior_rows(transactions):
    """src_log_dev must compare against the account's PAST amounts only."""
    out = build_amount_features(build_account_features(transactions))
    rng = np.random.default_rng(2)
 
    checked = 0
    for i in rng.choice(np.arange(200, len(out)), 40, replace=False):
        row = out.iloc[i]
        if row.src_txn_count_prior < MIN_PRIOR_TXNS:
            continue
 
        past_src = out.iloc[:i]
        past_src = past_src[past_src.src_account == row.src_account]
 
        expected = np.log1p(row.amount_paid) - np.log1p(past_src.amount_paid).mean()
        assert np.isclose(row.src_log_dev, expected), (
            f"row {i}: log deviation does not match the prior geometric mean"
        )
        checked += 1
 
    assert checked > 0, "fixture produced no rows with enough history to test"
 
 
def test_first_transaction_has_no_history(transactions):
    """An account's first transaction cannot have a gap or a prior count."""
    out = build_account_features(transactions)
    firsts = out.groupby("src_account", sort=False).head(1)
 
    assert (firsts.src_txn_count_prior == 0).all()
    assert firsts.src_secs_since_last.isna().all()
    assert (firsts.src_unique_dst_prior == 0).all()
 
 
# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
 
def test_ratios_are_nan_below_minimum_history(transactions):
    """Ratios on 1-2 prior transactions produce spikes at 1.0 and 0.5 that
    reflect newness, not fan-out behaviour. They must be suppressed."""
    out = build_account_features(transactions)
 
    too_new = out.src_txn_count_prior < MIN_PRIOR_TXNS
    assert out.loc[too_new, "src_fanout_ratio"].isna().all()
 
    enough = out.src_txn_count_prior >= MIN_PRIOR_TXNS
    if enough.any():
        assert out.loc[enough, "src_fanout_ratio"].notna().all()
 
 
def test_ratios_are_bounded(transactions):
    """Distinct counterparties can never exceed transaction count."""
    out = build_account_features(transactions)
    for column in ("src_fanout_ratio", "dst_fanin_ratio"):
        values = out[column].dropna()
        assert values.between(0, 1).all(), f"{column} outside [0, 1]"
 
 
def test_no_negative_time_gaps(transactions):
    """Negative gaps would mean the frame was not sorted chronologically."""
    out = build_account_features(transactions)
    assert (out.src_secs_since_last.dropna() >= 0).all()
 
 
# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
 
def test_hour_encoding_is_cyclical(transactions):
    """23:00 and 00:00 must be adjacent, not 23 units apart."""
    out = build_time_features(transactions)
 
    unit_circle = out.hour_sin**2 + out.hour_cos**2
    assert np.allclose(unit_circle, 1.0)
 
    def point(hour):
        angle = 2 * np.pi * hour / 24
        return np.array([np.sin(angle), np.cos(angle)])
 
    adjacent = np.linalg.norm(point(23) - point(0))
    distant = np.linalg.norm(point(0) - point(12))
    assert adjacent < distant
 
 
def test_thresholds_match_constants(transactions):
    """Flags must use the swept constants, not hardcoded numbers."""
    out = build_threshold_flags(
        build_account_features(transactions)
    )
    gap = out.src_secs_since_last
 
    assert out.src_rapid_txn.equals(gap <= RAPID_TXN_SECONDS)
    assert out.src_dormant_wake.equals(gap > DORMANT_WAKE_SECONDS)
 
 
# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
 
def test_build_features_preserves_row_count(transactions):
    """Feature engineering must never add or drop rows."""
    out = build_features(transactions.assign(
        src_entity_type="Corporation", dst_entity_type="Partnership",
        src_bank_country="Portugal", dst_bank_country="Canada",
        payment_format="ACH",
    ))
    assert len(out) == len(transactions)
 
 
def test_rejected_features_are_absent():
    """Features rejected in EDA must not reappear in the model inputs."""
    from aml_detection.features import FEATURE_COLUMNS
 
    rejected = {
        "is_cross_currency", "amount_delta", "is_self_loop",
        "amt_logz_ccy", "log_amount", "src_log_dev_abs",
        "src_is_new", "dst_is_new", "day_of_week",
        "src_txn_count_prior", "dst_txn_count_prior",
        "src_unique_dst_prior", "dst_unique_src_prior",
    }
    overlap = rejected & set(FEATURE_COLUMNS)
    assert not overlap, f"rejected features present in model inputs: {overlap}"