from __future__ import annotations
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
 
TARGET = "is_laundering"
 
#: Plots on millions of rows are slow and no more informative than a sample.
PLOT_SAMPLE = 200_000
 
 
# --------------------------------------------------------------------------
# dataset level
# --------------------------------------------------------------------------
 
def overview(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """One row per column: dtype, missingness, cardinality, sample values.
 
    The first thing to run. Identifies constant columns, high-cardinality
    identifiers, and unexpected nulls before any modelling decision is made.
    """
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "dtype": str(series.dtype),
                "nulls": series.isna().sum(),
                "null_pct": series.isna().mean(),
                "unique": series.nunique(dropna=True),
                "unique_pct": series.nunique(dropna=True) / len(df),
                "example": series.dropna().iloc[0] if series.notna().any() else None,
            }
        )
    summary = pd.DataFrame(rows, index=df.columns)
 
    constant = summary.query("unique <= 1").index.tolist()
    if constant:
        logger.warning(f"Constant columns (no information): {constant}")
 
    identifier_like = summary.query("unique_pct > 0.9").index.tolist()
    if identifier_like:
        logger.warning(
            f"Near-unique columns, likely identifiers not features: {identifier_like}"
        )
    return summary
 
 
def target_balance(df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Class balance, and what it implies for metric choice."""
    positives = int(df[target].sum())
    rate = df[target].mean()
 
    logger.info(
        f"{positives:,} positives in {len(df):,} rows ({rate:.4%}). "
        f"1 in {int(1 / rate):,}."
    )
    if rate < 0.01:
        logger.warning(
            "Severe imbalance: ROC-AUC will look flattering and mean little. "
            "Report PR-AUC and recall at a fixed alert budget instead."
        )
    return pd.Series({"rows": len(df), "positives": positives, "rate": rate})
 
 
# --------------------------------------------------------------------------
# categorical
# --------------------------------------------------------------------------
 
def explore_categorical(
    df: pd.DataFrame,
    column: str,
    target: str = TARGET,
    min_group: int = 500,
    top_n: int = 15,
    plot: bool = True,
) -> pd.DataFrame:
    """Frequency, target rate, and lift for each category.
 
    `lift` is the category's target rate divided by the overall rate, so 1.0
    means the category tells you nothing. Groups smaller than `min_group` are
    excluded because a couple of positives in a rare category produces a large
    lift that is pure noise.
    """
    overall = df[target].mean()
 
    stats = (
        df.groupby(column, observed=True)[target]
        .agg(n="size", positives="sum", rate="mean")
        .assign(
            share=lambda d: d["n"] / len(df),
            lift=lambda d: d["rate"] / overall,
        )
        .sort_values("positives", ascending=False)
    )
 
    small = stats.query("n < @min_group")
    if len(small):
        logger.info(f"{len(small)} categories below {min_group} rows excluded from plot")
    plotted = stats.query("n >= @min_group").head(top_n)
 
    if plot and len(plotted):
        fig, axes = plt.subplots(1, 3, figsize=(16, max(3.5, 0.35 * len(plotted))))
 
        plotted["n"].plot.barh(ax=axes[0], color="#4a6fa5")
        axes[0].set_title(f"{column}: row count")
        axes[0].set_xlabel("rows")
 
        plotted["positives"].plot.barh(ax=axes[1], color="#e07a00")
        axes[1].set_title("positives (absolute)")
        axes[1].set_xlabel("count")
 
        colors = ["#e63946" if v > 1 else "#adb5bd" for v in plotted["lift"]]
        plotted["lift"].plot.barh(ax=axes[2], color=colors)
        axes[2].axvline(1.0, color="black", ls="--", lw=1)
        axes[2].set_title("lift vs overall rate (1.0 = no signal)")
        axes[2].set_xlabel("lift")
 
        for ax in axes:
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
 
    return stats
 
 
# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------
 
def explore_numeric(
    df: pd.DataFrame,
    column: str,
    target: str = TARGET,
    log_scale: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """Distribution of a numeric column, split by target class.
 
    Financial amounts are heavily right-skewed, so the default is a log x-axis;
    on a linear axis every distribution collapses into a spike at zero with an
    invisible tail. `log_scale=False` for columns that are not amounts.
    """
    stats = (
        df.groupby(target)[column]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        .T
    )
 
    if plot:
        sample = df if len(df) <= PLOT_SAMPLE else df.sample(PLOT_SAMPLE, random_state=42)
        negative = sample.loc[sample[target] == 0, column].dropna()
        positive = sample.loc[sample[target] == 1, column].dropna()
 
        # keep all positives: they are rare and sampling can leave almost none
        positive_full = df.loc[df[target] == 1, column].dropna()
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
 
        if log_scale:
            valid_neg = negative[negative > 0]
            valid_pos = positive_full[positive_full > 0]
            bins = np.logspace(
                np.log10(max(valid_neg.min(), 1e-6)),
                np.log10(valid_neg.max()),
                60,
            )
            axes[0].set_xscale("log")
        else:
            valid_neg, valid_pos = negative, positive_full
            bins = 60
 
        axes[0].hist(valid_neg, bins=bins, alpha=0.6, density=True,
                     label="normal", color="#4a6fa5")
        if len(valid_pos):
            axes[0].hist(valid_pos, bins=bins, alpha=0.6, density=True,
                         label="laundering", color="#e63946")
        else:
            logger.warning(f"No positive rows with {column} > 0 to plot")
        axes[0].set_title(f"{column}: distribution by class (density)")
        axes[0].set_xlabel(column)
        axes[0].legend()
 
        # decile view: does the target rate move monotonically with the value?
        deciles = pd.qcut(df[column], 10, duplicates="drop")
        rate_by_decile = df.groupby(deciles, observed=True)[target].mean()
        rate_by_decile.plot.bar(ax=axes[1], color="#e07a00")
        axes[1].axhline(df[target].mean(), color="black", ls="--", lw=1,
                        label="overall rate")
        axes[1].set_title(f"target rate by {column} decile")
        axes[1].set_ylabel("laundering rate")
        axes[1].tick_params(axis="x", rotation=90, labelsize=7)
        axes[1].legend()
 
        plt.tight_layout()
        plt.show()
 
    return stats
 
 
# --------------------------------------------------------------------------
# boolean
# --------------------------------------------------------------------------
 
def explore_boolean(
    df: pd.DataFrame,
    column: str,
    target: str = TARGET,
    plot: bool = True,
) -> pd.DataFrame:
    """Target rate for a True/False column, with lift.
 
    A flag whose lift is ~1.0 on both values carries no signal and should be
    dropped rather than left for the model to ignore.
    """
    overall = df[target].mean()
    stats = (
        df.groupby(column, observed=True)[target]
        .agg(n="size", positives="sum", rate="mean")
        .assign(lift=lambda d: d["rate"] / overall)
    )
 
    if stats["positives"].eq(0).any():
        empty = stats.query("positives == 0").index.tolist()
        logger.warning(
            f"{column}={empty} contains ZERO positives. Either the flag is "
            "useless, or those rows can be excluded from training entirely."
        )
 
    if plot:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        colors = ["#e63946" if v > 1 else "#adb5bd" for v in stats["lift"]]
        stats["rate"].plot.bar(ax=ax, color=colors)
        ax.axhline(overall, color="black", ls="--", lw=1, label="overall rate")
        ax.set_title(f"{column}: laundering rate")
        ax.set_ylabel("rate")
        ax.legend()
        plt.tight_layout()
        plt.show()
 
    return stats
 
 
# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------
 
def explore_temporal(
    df: pd.DataFrame,
    column: str = "timestamp",
    target: str = TARGET,
    plot: bool = True,
) -> pd.DataFrame:
    """Volume and target rate by hour of day and day of week.
 
    Hour-of-day and day-of-week are safe features: a real bank knows both at
    scoring time. Absolute date or "days since file start" is NOT safe, because
    it encodes position within this particular file.
    """
    frame = df.assign(
        hour=lambda d: d[column].dt.hour,
        dayofweek=lambda d: d[column].dt.dayofweek,
    )
 
    by_hour = frame.groupby("hour")[target].agg(n="size", positives="sum", rate="mean")
    by_dow = frame.groupby("dayofweek")[target].agg(n="size", positives="sum", rate="mean")
 
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 6.5))
        overall = df[target].mean()
 
        by_hour["n"].plot.bar(ax=axes[0, 0], color="#4a6fa5")
        axes[0, 0].set_title("volume by hour of day")
 
        by_hour["rate"].plot.bar(ax=axes[0, 1], color="#e63946")
        axes[0, 1].axhline(overall, color="black", ls="--", lw=1)
        axes[0, 1].set_title("laundering rate by hour")
 
        by_dow["n"].plot.bar(ax=axes[1, 0], color="#4a6fa5")
        axes[1, 0].set_title("volume by day of week (0=Mon)")
 
        by_dow["rate"].plot.bar(ax=axes[1, 1], color="#e63946")
        axes[1, 1].axhline(overall, color="black", ls="--", lw=1)
        axes[1, 1].set_title("laundering rate by day of week")
 
        plt.tight_layout()
        plt.show()
 
    return pd.concat({"hour": by_hour, "dayofweek": by_dow}, names=["unit"])