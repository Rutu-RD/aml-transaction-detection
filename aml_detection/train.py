"""Model training for AML detection.

WHAT THIS COMPARES
------------------
Three model families, each with two imbalance strategies, all measured against
the rules baseline (PR-AUC 0.0126 on validation).

    LogisticRegression   interpretable reference. Expected to underperform:
                         the strongest features are non-monotonic (amount is a
                         band, timing is U-shaped, fan-in is a threshold) and a
                         linear model cannot represent "both extremes are
                         risky".

    RandomForest         handles non-monotonic relationships, no gradient
                         boosting. Sits between the other two.

    XGBoost              primary candidate. Handles NaN natively, which
                         matters here because 26% of dst_fanin_ratio is
                         missing by design.

IMBALANCE
---------
Two strategies, reported honestly whichever wins:

    class weighting   scale_pos_weight (XGB) or class_weight='balanced'.
                      Reweights the loss without inventing data.

    SMOTE             synthesises minority examples by interpolating between
                      neighbours. Applied to TRAINING FOLDS ONLY -- generating
                      synthetic positives from validation neighbours is a
                      classic and invisible leak.

Prior expectation is that class weighting wins: SMOTE interpolates in feature
space, and several features here are ratios and flags where the midpoint
between two real transactions is not a plausible transaction. Reporting a
negative SMOTE result is more credible than the usual "applied SMOTE, got
great numbers".

METRICS
-------
PR-AUC is the headline. At a 0.11% positive rate ROC-AUC is flattering and
close to uninformative -- a model can score 0.95 while being useless at the
only threshold anyone would operate at.

recall@k answers the question an AML function actually faces: investigator
capacity is fixed, so of the k alerts we can review per period, how much
laundering do we catch?

Everything is logged to MLflow so runs are comparable and the winning model is
reproducible from its run id.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aml_detection.config import MODELS_DIR, REPORTS_DIR
from aml_detection.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    feature_matrix,
)
from aml_detection.splitting import load_split

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

RANDOM_STATE = 42

#: Alert volumes at which recall is reported. Chosen to bracket plausible
#: investigator capacity for a mid-size institution.
K_VALUES: tuple[int, ...] = (500, 1_000, 5_000, 10_000)

#: Rules baseline on validation, from notebooks/02_baseline.ipynb. Every model
#: is reported as a multiple of this rather than as a standalone number.
BASELINE_PR_AUC = 0.0126


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def build_preprocessor(handles_nan: bool = False) -> ColumnTransformer:
    """Fitted transformations for the three feature families.

    handles_nan
        True for models with native missing-value support (XGBoost).
        Imputation is skipped so the model can learn which side of a split a
        missing value belongs on -- that is information, not noise.

        False for models that cannot accept NaN. Median imputation plus an
        indicator column, so "no history available" stays visible rather than
        being disguised as a median-valued account.

    Why NaN is not simply filled with zero: dst_fanin_ratio and src_log_dev are
    suppressed below three prior transactions (1/1 and 1/2 produce spikes at
    1.0 and 0.5 that reflect newness, not fan-out). Filling with 0 would assert
    "zero counterparty diversity", which is false and materially different from
    "unknown".
    """
    numeric_steps = []
    if not handles_nan:
        numeric_steps.append(
            ("impute", SimpleImputer(strategy="median", add_indicator=True))
        )
    numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # handle_unknown="ignore": a bank country unseen in training must not
        # crash inference. min_frequency collapses rare categories.
        ("encode", OneHotEncoder(
            handle_unknown="ignore", min_frequency=0.01, sparse_output=False
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("bool", "passthrough", list(BOOLEAN_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",   # nothing outside FEATURE_COLUMNS can reach the model
    )


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """One model configuration to train and evaluate."""

    name: str
    estimator: object
    handles_nan: bool = False
    needs_smote: bool = False
    params: dict = field(default_factory=dict)


def build_model_specs(scale_pos_weight: float) -> list[ModelSpec]:
    """Every model and imbalance strategy to compare.

    scale_pos_weight is negatives/positives (~709 here). It tells the loss
    function to treat one missed positive as costing as much as ~709 false
    positives, which approximates the real cost asymmetry: a missed STR is a
    regulatory failure, a false positive costs investigator review time.
    """
    specs = [
        ModelSpec(
            name="logreg_weighted",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            params={"class_weight": "balanced", "max_iter": 1_000},
        ),
        ModelSpec(
            name="random_forest_weighted",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=50,      # guards against memorising rare rows
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            params={"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 50},
        ),
    ]

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed -- skipping the primary candidate")
        return specs

    specs.append(
        ModelSpec(
            name="xgboost_weighted",
            estimator=XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",      # optimise the metric we report
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            handles_nan=True,
            params={
                "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                "scale_pos_weight": round(scale_pos_weight, 1),
            },
        )
    )

    # Same model, imbalance handled by resampling instead of reweighting.
    specs.append(
        ModelSpec(
            name="xgboost_smote",
            estimator=XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            handles_nan=False,   # SMOTE cannot interpolate across NaN
            needs_smote=True,
            params={"n_estimators": 400, "max_depth": 6, "sampling": "SMOTE"},
        )
    )
    return specs


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Performance if investigators can review exactly k alerts.

    This is the operating question in an AML function: capacity is fixed, so
    "what is the AUC" matters less than "of the k cases we can review, how many
    are real and what share of laundering do we catch?"
    """
    k = min(k, len(scores))
    top_k = np.argpartition(-scores, k - 1)[:k]
    caught = int(y_true[top_k].sum())
    total = int(y_true.sum())

    return {
        f"precision_at_{k}": caught / k,
        f"recall_at_{k}": caught / total if total else np.nan,
        f"lift_at_{k}": (caught / k) / y_true.mean() if y_true.mean() else np.nan,
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Every metric reported for a run.

    ROC-AUC is included for reference only. At a 0.11% positive rate it is
    dominated by the vast majority of easy negatives and can look excellent
    while the model is useless at any usable threshold.
    """
    metrics = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "base_rate": float(y_true.mean()),
    }
    metrics["pr_auc_vs_baseline"] = metrics["pr_auc"] / BASELINE_PR_AUC
    metrics["pr_auc_vs_random"] = metrics["pr_auc"] / metrics["base_rate"]

    for k in K_VALUES:
        metrics.update(recall_at_k(y_true, scores, k))
    return metrics


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_one(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict]:
    """Fit one specification and score it on validation.

    Preprocessing is inside the Pipeline rather than applied separately, so the
    fitted transformations travel with the estimator when it is pickled.
    Inference then reapplies exactly what training produced instead of
    recomputing statistics on incoming data -- the primary train-serve skew
    defence.
    """
    logger.info(f"Training {spec.name}")

    steps = [("preprocess", build_preprocessor(handles_nan=spec.handles_nan))]

    if spec.needs_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError:
            logger.warning(f"imbalanced-learn missing -- skipping {spec.name}")
            return None, {}

        # ImbPipeline applies SMOTE during fit only, never during predict, so
        # synthetic rows cannot leak into the validation score.
        pipeline = ImbPipeline(
            steps + [
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                ("model", spec.estimator),
            ]
        )
    else:
        pipeline = Pipeline(steps + [("model", spec.estimator)])

    pipeline.fit(X_train, y_train)
    scores = pipeline.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val.to_numpy(), scores)

    logger.success(
        f"{spec.name}: PR-AUC {metrics['pr_auc']:.4f} "
        f"({metrics['pr_auc_vs_baseline']:.1f}x baseline) | "
        f"recall@1000 {metrics['recall_at_1000']:.3f}"
    )
    return pipeline, metrics


def leakage_check(pipeline: Pipeline, spec: ModelSpec) -> pd.Series:
    """Feature importance concentration, as a leakage smoke test.

    A single feature carrying most of the model's importance usually means it
    encodes the answer rather than a behavioural signal. Healthy models spread
    importance across several features.

    This does not prove absence of leakage -- it catches the obvious case
    cheaply, so the expensive investigation only happens when warranted.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = (
        pd.Series(model.feature_importances_, index=names)
        .sort_values(ascending=False)
    )

    top_share = importance.iloc[0]
    logger.info(
        f"{spec.name} importance: top={importance.index[0]} ({top_share:.1%}), "
        f"top3={importance.head(3).sum():.1%}"
    )
    if top_share > 0.50:
        logger.warning(
            f"'{importance.index[0]}' carries {top_share:.1%} of importance. "
            "A single dominant feature is the usual signature of leakage -- "
            "verify it could be computed at scoring time before reporting."
        )
    return importance


def run_experiments(
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    use_mlflow: bool = True,
) -> pd.DataFrame:
    """Train every specification and return a comparison table."""
    train = load_split("train") if train is None else train
    val = load_split("val") if val is None else val

    X_train, y_train = feature_matrix(train), train[TARGET]
    X_val, y_val = feature_matrix(val), val[TARGET]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(
        f"train {len(X_train):,} rows, {int(y_train.sum()):,} positives "
        f"({y_train.mean():.4%}) | scale_pos_weight {scale_pos_weight:.0f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("aml_detection")
        except ImportError:
            logger.warning("mlflow not installed -- running without tracking")
            use_mlflow = False

    results, best = {}, (None, None, -1.0)

    for spec in build_model_specs(scale_pos_weight):
        if use_mlflow:
            import mlflow
            with mlflow.start_run(run_name=spec.name):
                mlflow.log_params(spec.params)
                mlflow.log_param("n_features", len(FEATURE_COLUMNS))
                mlflow.log_param("train_positives", int(y_train.sum()))

                pipeline, metrics = train_one(spec, X_train, y_train, X_val, y_val)
                if pipeline is None:
                    continue

                importance = leakage_check(pipeline, spec)
                if len(importance):
                    importance.head(20).to_csv(
                        REPORTS_DIR / f"importance_{spec.name}.csv"
                    )
                    mlflow.log_metric("top_feature_share", float(importance.iloc[0]))

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    pipeline, name="model",
                    serialization_format="cloudpickle",
                )
        else:
            pipeline, metrics = train_one(spec, X_train, y_train, X_val, y_val)
            if pipeline is None:
                continue

            importance = leakage_check(pipeline, spec)
            if len(importance):
                importance.head(20).to_csv(
                    REPORTS_DIR / f"importance_{spec.name}.csv"
                )

        results[spec.name] = metrics
        if metrics["pr_auc"] > best[2]:
            best = (spec.name, pipeline, metrics["pr_auc"])

    comparison = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)

    logger.success(f"Best: {best[0]} with PR-AUC {best[2]:.4f}")
    if best[2] < BASELINE_PR_AUC:
        logger.error(
            f"No model beat the rules baseline ({BASELINE_PR_AUC:.4f}). "
            "Investigate before proceeding -- either the features carry less "
            "signal than the rules encode, or something is wrong upstream."
        )
    elif best[2] > 0.30:
        logger.warning(
            f"PR-AUC {best[2]:.4f} is unusually high for a 0.11% positive "
            "rate. Check for leakage before reporting this."
        )

    return comparison


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

def main() -> None:
    comparison = run_experiments()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")

    headline = comparison.iloc[0]
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "best_model": comparison.index[0],
                "pr_auc": float(headline["pr_auc"]),
                "pr_auc_vs_baseline": float(headline["pr_auc_vs_baseline"]),
                "recall_at_1000": float(headline["recall_at_1000"]),
                "baseline_pr_auc": BASELINE_PR_AUC,
            },
            indent=2,
        )
    )
    logger.info("\n" + comparison.round(4).to_string())


if __name__ == "__main__":
    main()