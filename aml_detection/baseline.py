"""Rules-based baseline for AML detection.
 
WHY THIS EXISTS
---------------
Every real AML function already runs on rules. A model is only worth deploying
if it beats the rules that are already in production, so "XGBoost achieved
0.40 PR-AUC" is not a result -- it is a number without a reference point.
"rules achieve 0.12, the model achieves 0.40 at the same alert volume" is a
business case.
 
This module implements the incumbent. Every rule below comes from a measured
finding in notebooks/01_eda.ipynb, not from intuition.
 
HOW RULES DIFFER FROM A MODEL
-----------------------------
Rules are deterministic and independently auditable: an investigator can read
"amount in the structuring band AND moved within a minute" and verify it by
hand. That is why they persist in regulated environments despite worse
performance.
 
Their weakness is that each rule fires independently. They cannot express "high
fan-in matters only when the amount is also in the band", which is exactly the
kind of interaction a tree ensemble captures. The gap between the two is the
value the model adds.
 
SCORING
-------
Two modes, because they answer different questions:
 
    any-rule    fires if ANY rule matches. Maximises recall, floods the queue.
                This is what an unoptimised rules deployment looks like.
 
    rule count  number of rules matched, 0-N. Gives a crude ranking, which
                makes precision@k and PR-AUC computable and therefore
                comparable against the model.
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
from loguru import logger
 
from aml_detection.features import (
    DORMANT_WAKE_SECONDS,
    HIGH_FANIN_RATIO,
    RAPID_TXN_SECONDS,
)
 
TARGET = "is_laundering"
 
# --------------------------------------------------------------------------
# Rule thresholds
# --------------------------------------------------------------------------
 
#: Laundering concentrates in an amount band rather than at the top end.
#: Percentile rank within currency: rate climbs to 0.0022 at decile 9 then
#: COLLAPSES to 0.0003 in decile 10. Large enough to be worth moving, small
#: enough to avoid the scrutiny the largest transactions attract.
STRUCTURING_BAND: tuple[float, float] = (0.55, 0.90)
 
#: ACH carries 83% of positives from 11% of rows (lift 7.3x).
HIGH_RISK_FORMATS: tuple[str, ...] = ("ACH",)
 
#: Business hours, where laundering blends into volume. Rate ~0.0013 versus
#: ~0.0002 at hour 0.
BUSINESS_HOURS: tuple[int, int] = (11, 16)
 
 
# --------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------
# Each returns a boolean Series and is measured separately, so a rule that
# fires often but adds no precision can be identified and removed.
 
def rule_high_risk_format(df: pd.DataFrame) -> pd.Series:
    """Channel risk. ACH holds the overwhelming majority of positives."""
    return df["payment_format"].isin(HIGH_RISK_FORMATS)
 
 
def rule_structuring_band(df: pd.DataFrame) -> pd.Series:
    """Amount sits in the band laundering favours, relative to its currency.
 
    Uses percentile-within-currency rather than an absolute threshold, because
    currency medians span six orders of magnitude and the file has no exchange
    rates. An absolute "under 10,000" rule would be meaningless across Bitcoin,
    Rupee and US Dollar simultaneously.
    """
    low, high = STRUCTURING_BAND
    return df["amt_pct_ccy"].between(low, high)
 
 
def rule_rapid_passthrough(df: pd.DataFrame) -> pd.Series:
    """Money leaves almost immediately after the account's previous activity.
 
    The account is a conduit rather than a destination -- the layering step.
    """
    return df["src_secs_since_last"] <= RAPID_TXN_SECONDS
 
 
def rule_dormant_reactivation(df: pd.DataFrame) -> pd.Series:
    """A quiet account suddenly moves money. Classic mule behaviour."""
    return df["src_secs_since_last"] > DORMANT_WAKE_SECONDS
 
 
def rule_high_fanin(df: pd.DataFrame) -> pd.Series:
    """Nearly every incoming payment comes from a different sender.
 
    The gather step: a collection account receiving from many mules. Normal
    accounts receive repeatedly from the same few sources.
    """
    return df["dst_fanin_ratio"] > HIGH_FANIN_RATIO
 
 
def rule_amount_spike(df: pd.DataFrame) -> pd.Series:
    """Transaction far above this account's own typical amount.
 
    Threshold of 2.0 in log space is roughly a 7x multiple of the account's
    geometric mean.
    """
    return df["src_log_dev"] > 2.0
 
 
def rule_business_hours(df: pd.DataFrame) -> pd.Series:
    """Sent during the window where laundering blends into peak volume."""
    start, end = BUSINESS_HOURS
    return df["hour"].between(start, end)
 
 
RULES: dict[str, callable] = {
    "high_risk_format": rule_high_risk_format,
    "structuring_band": rule_structuring_band,
    "rapid_passthrough": rule_rapid_passthrough,
    "dormant_reactivation": rule_dormant_reactivation,
    "high_fanin": rule_high_fanin,
    "amount_spike": rule_amount_spike,
    "business_hours": rule_business_hours,
}
 
 
# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
 
def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every rule, returning one boolean column per rule.
 
    NaN handling: a rule cannot fire on missing data. dst_fanin_ratio and
    src_log_dev are NaN for accounts with fewer than three prior transactions,
    and those rows are treated as not-flagged rather than flagged. Real systems
    behave the same way -- an unknown does not raise an alert.
    """
    flags = {name: rule(df).fillna(False) for name, rule in RULES.items()}
    return pd.DataFrame(flags, index=df.index)
 
 
def rule_scores(df: pd.DataFrame) -> pd.Series:
    """Number of rules matched per transaction.
 
    A crude ranking, but enough to make PR-AUC and precision@k computable and
    therefore directly comparable to the model's scores.
    """
    return apply_rules(df).sum(axis=1)
 
 
def any_rule_fires(df: pd.DataFrame) -> pd.Series:
    """Binary alert: at least one rule matched."""
    return apply_rules(df).any(axis=1)
 
 
# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
 
def evaluate_individual_rules(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Precision, recall and lift for each rule in isolation.
 
    A rule that fires on 40% of transactions to catch 45% of laundering is
    barely better than random and is costing investigator time for nothing.
    Lift near 1.0 is the tell.
    """
    flags = apply_rules(df)
    labels = df[target]
    base_rate = labels.mean()
    total_positives = labels.sum()
 
    rows = {}
    for name in flags.columns:
        fired = flags[name]
        caught = int((fired & (labels == 1)).sum())
        n_fired = int(fired.sum())
 
        rows[name] = {
            "fired": n_fired,
            "fired_pct": n_fired / len(df),
            "caught": caught,
            "precision": caught / n_fired if n_fired else np.nan,
            "recall": caught / total_positives if total_positives else np.nan,
            "lift": (caught / n_fired) / base_rate if n_fired else np.nan,
        }
 
    return pd.DataFrame(rows).T.sort_values("lift", ascending=False)
 
 
def evaluate_baseline(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Performance at each possible rule-count threshold.
 
    Reading this table: as the threshold rises, precision improves and recall
    falls. The row an AML team cares about is whichever threshold produces an
    alert volume their investigators can actually review.
    """
    scores = rule_scores(df)
    labels = df[target]
    total_positives = labels.sum()
    base_rate = labels.mean()
 
    rows = {}
    for threshold in range(1, len(RULES) + 1):
        fired = scores >= threshold
        n_fired = int(fired.sum())
        caught = int((fired & (labels == 1)).sum())
 
        rows[f">={threshold} rules"] = {
            "alerts": n_fired,
            "alert_rate": n_fired / len(df),
            "caught": caught,
            "precision": caught / n_fired if n_fired else np.nan,
            "recall": caught / total_positives if total_positives else np.nan,
            "lift": (caught / n_fired) / base_rate if n_fired else np.nan,
        }
 
    return pd.DataFrame(rows).T
 
 
def precision_at_k(df: pd.DataFrame, k: int, target: str = TARGET) -> dict:
    """What an investigator team reviewing k alerts per period would find.
 
    This is the metric an AML function actually operates on: capacity is fixed,
    so the question is not "what is the AUC" but "of the k cases we can review,
    how many are real, and what share of laundering do we catch?"
 
    Ties are broken randomly with a fixed seed, since rule counts are coarse
    and many transactions share the same score.
    """
    scores = rule_scores(df)
    labels = df[target]
 
    rng = np.random.default_rng(42)
    order = pd.Series(scores + rng.random(len(scores)) * 0.5, index=df.index)
    top_k = order.nlargest(k).index
 
    caught = int(labels.loc[top_k].sum())
    return {
        "k": k,
        "caught": caught,
        "precision_at_k": caught / k,
        "recall_at_k": caught / labels.sum() if labels.sum() else np.nan,
        "lift": (caught / k) / labels.mean() if labels.mean() else np.nan,
    }
 
 
def baseline_report(df: pd.DataFrame, target: str = TARGET) -> dict[str, pd.DataFrame]:
    """Full baseline evaluation, for the model to be measured against."""
    logger.info(f"Evaluating rules baseline on {len(df):,} rows")
 
    individual = evaluate_individual_rules(df, target)
    combined = evaluate_baseline(df, target)
    at_k = pd.DataFrame(
        [precision_at_k(df, k, target) for k in (100, 500, 1_000, 5_000, 10_000)]
    ).set_index("k")
 
    weak = individual.query("lift < 1.2").index.tolist()
    if weak:
        logger.warning(
            f"Rules with lift below 1.2 are barely better than random and "
            f"cost investigator time for little return: {weak}"
        )
 
    return {"individual": individual, "combined": combined, "at_k": at_k}