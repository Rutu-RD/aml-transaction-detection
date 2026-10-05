"""Account-network features from the transaction graph.

WHY
---
The dataset is already a graph: every row has a sender, a receiver and a
timestamp. The existing 16 features read it row by row, which makes three
typologies invisible:

    collection / mule   many senders -> one account -> one large payment out.
                        Each inbound payment is unremarkable on its own.
    layering chain      A->B->C->D, similar amounts, one hop per day.
                        Each hop looks like a normal transfer.
    round trip          A->B->C->A. No single row closes the loop.

These only appear when you look across rows, which is what this module does.

THE LEAKAGE PROBLEM, AND THE RULE
---------------------------------
Given:

    Day 1:  A -> B
    Day 1:  C -> B
    Day 2:  D -> B

B's in-degree is 2 on day 1 and 3 on day 2. Computing it once over the finished
table gives 3 everywhere, so a day-1 transaction gets scored using a day-2
payment. That inflates PR-AUC convincingly and is invisible in the output.

The rule enforced throughout: **every feature comes from a window that ends
strictly before the transaction it scores.** Implemented as a half-open
interval [day - W, day), and asserted by tests/test_graph_features.py, which
deletes every edge at or after t, recomputes, and requires bit-identical
output.

WHY MULTIPLE WINDOW WIDTHS
--------------------------
A 1-day window sees bursts; a 7-day window sees baseline behaviour. The ratio
between them encodes acceleration, which is often more informative than either
alone -- an account whose 1-day degree approaches its 7-day degree is doing
something new.

The dataset spans 10 days, so windows are chosen against that: 30 days would
swallow the file and produce a constant.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Community-level label density -- the fraction of flagged accounts in your
graph community. It carries the highest signal of anything in this family and
is the fastest way to leak labels, because it uses the target of neighbouring
rows. It is only legitimate with labels confirmed before T and a realistic
investigation lag modelled in, neither of which this dataset supports.

Structural features needing igraph (PageRank, k-core, cycle counts) are also
excluded from this version. They are only worth the dependency if the degree
and flow features below move the number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from aml_detection.config import PROCESSED_DATA_DIR

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Window widths in days. The file spans 10 days, so these are chosen to fit:
#: 1d catches bursts, 7d establishes a baseline, 3d sits between.
WINDOW_DAYS: tuple[int, ...] = (1, 3, 7)

#: Features that are counts. A node with no prior history genuinely has zero
#: counterparties, so filling with 0 states a fact.
COUNT_FEATURES: tuple[str, ...] = (
    "in_degree", "out_degree", "txn_count_in", "txn_count_out",
)

#: Features that are ratios. A node with no prior history has an UNDEFINED
#: ratio, not a zero one. Filling these with 0 would conflate "never funded"
#: with "received money and kept all of it", which are opposite behaviours.
RATIO_FEATURES: tuple[str, ...] = (
    "fan_ratio", "flow_through", "amt_concentration", "reciprocity",
)

GRAPH_FEATURES_FILE = "transactions_graph_features.parquet"


# --------------------------------------------------------------------------
# Node-level features for one window
# --------------------------------------------------------------------------

def _node_features(window: pd.DataFrame) -> pd.DataFrame:
    """Degree and flow features for every account active in one window.

    Pure groupby -- no graph library needed for this tier. The only non-obvious
    computation is reciprocity, handled separately below.
    """
    if window.empty:
        return pd.DataFrame()

    incoming = window.groupby("dst_account", observed=True).agg(
        in_degree=("src_account", "nunique"),
        txn_count_in=("amount_paid", "size"),
        amt_in_sum=("amount_paid", "sum"),
        amt_in_max=("amount_paid", "max"),
        unique_currencies_in=("currency_paid", "nunique"),
        unique_formats_in=("payment_format", "nunique"),
    )

    outgoing = window.groupby("src_account", observed=True).agg(
        out_degree=("dst_account", "nunique"),
        txn_count_out=("amount_paid", "size"),
        amt_out_sum=("amount_paid", "sum"),
    )

    nodes = incoming.join(outgoing, how="outer").fillna(0)

    # High in-degree against low out-degree is the collection/mule shape.
    # +1 in the denominator keeps it finite for accounts that only receive.
    nodes["fan_ratio"] = nodes["in_degree"] / (nodes["out_degree"] + 1)

    # Pass-through accounts sit near 1.0: almost everything that arrived left
    # again. A savings account sits near 0.
    nodes["flow_through"] = np.where(
        nodes["amt_in_sum"] > 0,
        nodes["amt_out_sum"] / (nodes["amt_in_sum"] + 1),
        np.nan,          # nothing came in -- undefined, not zero
    )

    # One large deposit versus many small ones. Structuring pushes this down.
    nodes["amt_concentration"] = np.where(
        nodes["amt_in_sum"] > 0,
        nodes["amt_in_max"] / nodes["amt_in_sum"],
        np.nan,
    )

    return nodes.join(_reciprocity(window), how="left")


def _reciprocity(window: pd.DataFrame) -> pd.Series:
    """Fraction of an account's counterparties it both paid and was paid by.

    Ordinary commerce is often two-way: you pay a supplier, they refund you.
    A layering chain is strictly one-way, so it sits at 0. This is one of the
    few cheap features that separates a chain from a busy legitimate account
    with similar degree.

    Computed by building the set of directed pairs, then checking which have
    their reverse present.
    """
    pairs = window[["src_account", "dst_account"]].drop_duplicates()

    forward = set(zip(pairs["src_account"], pairs["dst_account"]))
    pairs = pairs.assign(
        is_reciprocal=[(d, s) in forward
                       for s, d in zip(pairs["src_account"], pairs["dst_account"])]
    )

    # Look at each pair from both endpoints, so a node sees every counterparty
    # regardless of which direction the money moved.
    endpoints = pd.concat([
        pairs.rename(columns={"src_account": "node", "dst_account": "counterparty"}),
        pairs.rename(columns={"dst_account": "node", "src_account": "counterparty"}),
    ])

    stats = endpoints.groupby("node", observed=True).agg(
        counterparties=("counterparty", "nunique"),
        reciprocal_edges=("is_reciprocal", "sum"),
    )

    # Each reciprocal relationship is counted twice (once per direction).
    reciprocity = (stats["reciprocal_edges"] / 2) / stats["counterparties"]
    return reciprocity.clip(0, 1).rename("reciprocity")


# --------------------------------------------------------------------------
# Rolling windows
# --------------------------------------------------------------------------

def build_window_table(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Node features for every (account, day), computed from the W days before.

    For a transaction on day D, the window is the half-open interval
    [D - W, D). Day D itself is excluded, which is the leakage defence: no
    transaction can contribute to the features used to score it, or to any
    other transaction on the same day.

    Day granularity is a deliberate simplification. Finer windows would let a
    transaction at 09:00 use one from 08:00 the same day, which is more
    information but also a much tighter leakage surface to verify. At a 10-day
    span, day-level is enough.
    """
    logger.info(f"Building {window_days}d window features")

    days = np.sort(df["day"].unique())
    frames = []

    for day in days:
        window = df[(df["day"] >= day - window_days) & (df["day"] < day)]
        if window.empty:
            continue

        nodes = _node_features(window)
        if nodes.empty:
            continue

        frames.append(nodes.assign(day=day).reset_index(names="account"))

    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    logger.info(
        f"  {len(table):,} (account, day) rows across {len(days)} days"
    )
    return table


# --------------------------------------------------------------------------
# Pair-level features
# --------------------------------------------------------------------------

def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """History of this specific sender-receiver relationship.

    is_first_interaction is the strongest feature in this tier. Ordinary
    payments repeat: the same landlord, the same supplier. A mule network is
    built from relationships that have never existed before.

    Uses the cumulative-count-minus-current identity so it stays prior-only
    without a per-group loop.
    """
    logger.info("Building pair-level features")

    df = df.sort_values("timestamp").reset_index(drop=True)
    pair = df.groupby(["src_account", "dst_account"], sort=False)

    df["pair_txn_count_prior"] = pair.cumcount()
    df["is_first_interaction"] = df["pair_txn_count_prior"] == 0

    # Deviation from this pair's own history. Log space because amounts are
    # multiplicative -- see features.build_amount_features for the reasoning.
    log_amount = np.log1p(df["amount_paid"])
    prior_mean = (
        (log_amount.groupby([df["src_account"], df["dst_account"]]).cumsum() - log_amount)
        / df["pair_txn_count_prior"].replace(0, np.nan)
    )
    df["pair_amt_log_dev"] = np.where(
        df["pair_txn_count_prior"] >= 2,
        log_amount - prior_mean,
        np.nan,
    )
    return df


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _join_side(
    df: pd.DataFrame,
    table: pd.DataFrame,
    side: str,
    window_days: int,
) -> pd.DataFrame:
    """Attach node features for one side of the transaction.

    Every transaction ends up carrying the sender's neighbourhood and the
    receiver's, so the feature count roughly doubles. Prefixed src_ / dst_ to
    keep them distinguishable, and suffixed with the window width.
    """
    feature_columns = [c for c in table.columns if c not in ("account", "day")]
    renamed = table.rename(
        columns={c: f"{side}_{c}_{window_days}d" for c in feature_columns}
    )

    before = len(df)
    merged = df.merge(
        renamed,
        left_on=[f"{side}_account", "day"],
        right_on=["account", "day"],
        how="left",              # keep every transaction, including day-1 rows
        validate="many_to_one",  # raises if the lookup key is not unique
    ).drop(columns="account")

    if len(merged) != before:
        raise ValueError(
            f"Row count changed joining {side} at {window_days}d: "
            f"{before:,} -> {len(merged):,}. Fan-out merge."
        )
    return merged


def _downcast(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Shrink the added columns to float32.

    At 5M rows, 85 extra float64 columns is roughly 3.4GB before the merges
    allocate their copies, which is enough to exhaust memory on a laptop.
    float32 halves it and costs nothing here: these are degree counts and
    bounded ratios, not quantities where the extra precision means anything.
    """
    for column in columns:
        if column in df.columns and df[column].dtype == "float64":
            df[column] = df[column].astype("float32")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill counts with zero, leave ratios null.

    An account with no prior window genuinely has zero counterparties, so 0 is
    a true statement for a count. It is not true for a ratio: flow_through of 0
    means "received money and sent none of it on", which is the opposite of
    "never received anything". XGBoost handles the nulls natively and can learn
    a separate branch for no-history accounts.
    """
    for column in df.columns:
        if any(column.startswith(f"{s}_{c}_") for s in ("src", "dst")
               for c in COUNT_FEATURES):
            df[column] = df[column].fillna(0)
    return df


def build_graph_features(
    df: pd.DataFrame,
    window_days: tuple[int, ...] = WINDOW_DAYS,
) -> pd.DataFrame:
    """Add node-level and pair-level network features.

    Must run on the full chronologically ordered frame, before splitting.
    Window logic depends on true chronology, and a per-split build would leave
    every validation and test row with no graph history.
    """
    logger.info(f"Building graph features on {len(df):,} rows")

    df = df.sort_values("timestamp").reset_index(drop=True)
    # Day index relative to the first transaction, so window arithmetic is
    # integer rather than timestamp comparison.
    df["day"] = (df["timestamp"] - df["timestamp"].min()).dt.days

    for width in window_days:
        table = build_window_table(df, width)
        if table.empty:
            logger.warning(f"No data for the {width}d window -- skipped")
            continue

        df = _join_side(df, table, "src", width)
        df = _join_side(df, table, "dst", width)
        df = _downcast(df, [c for c in df.columns if c.endswith(f"_{width}d")])
        del table

    df = add_pair_features(df)
    df = _fill_missing(df)

    # Acceleration: short-window activity relative to the longer baseline.
    # An account whose 1-day degree approaches its 7-day degree is doing
    # something it has not done before.
    if 1 in window_days and 7 in window_days:
        for side in ("src", "dst"):
            for metric in ("in_degree", "out_degree"):
                short = f"{side}_{metric}_1d"
                long = f"{side}_{metric}_7d"
                if short in df.columns and long in df.columns:
                    df[f"{side}_{metric}_accel"] = np.where(
                        df[long] > 0, df[short] / df[long], np.nan
                    )

    df = _downcast(df, [c for c in df.columns if c.endswith("_accel")
                        or c.startswith("pair_")])

    added = [c for c in df.columns if any(
        c.endswith(f"_{w}d") for w in window_days
    ) or c.endswith("_accel") or c.startswith("pair_") or c == "is_first_interaction"]
    logger.success(f"Added {len(added)} graph features")

    return df.drop(columns="day")


#: Graph features intended as model inputs. Kept separate from
#: features.FEATURE_COLUMNS so the two can be ablated independently -- the
#: point of this module is a single number: PR-AUC with, versus without.
GRAPH_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"{side}_{metric}_{w}d"
     for side in ("src", "dst")
     for metric in ("in_degree", "out_degree", "fan_ratio", "flow_through",
                    "reciprocity", "amt_concentration")
     for w in WINDOW_DAYS]
    + [f"{side}_{metric}_accel"
       for side in ("src", "dst")
       for metric in ("in_degree", "out_degree")]
    + ["pair_txn_count_prior", "is_first_interaction", "pair_amt_log_dev"]
)


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

def save_graph_features(df: pd.DataFrame, filename: str = GRAPH_FEATURES_FILE) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_parquet(path, index=False)
    logger.success(
        f"Wrote {len(df):,} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path


def main() -> None:
    source = PROCESSED_DATA_DIR / "transactions_features.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Run: python -m aml_detection.features"
        )

    df = pd.read_parquet(source)
    save_graph_features(build_graph_features(df))


if __name__ == "__main__":
    main()