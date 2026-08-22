"""Final evaluation on the held-out test split.
 
THIS MODULE READS THE TEST SET
------------------------------
Every other stage in this project uses train and validation only. Model
selection, hyperparameter search and threshold tuning all happened against
validation, so the numbers produced here are the first honest estimate of
performance on data no decision was made against.
 
That property is only worth something if this runs once. Re-running it, seeing
a disappointing number, changing something and re-running turns the test split
into a second validation set -- the estimate degrades quietly and there is no
warning when it does.
 
WHAT IS REPORTED
----------------
    headline        PR-AUC on test, and the multiple over the rules baseline.
 
    operating point threshold chosen on VALIDATION to hit a target recall, then
                    applied unchanged to test. Choosing the threshold on test
                    would inflate precision by fitting to the evaluation set.
 
    per segment     recall by payment format. The model draws ~54% of its
                    importance from ACH, so aggregate recall hides how it
                    behaves on other channels. An AML function needs to know
                    which typologies it is blind to.
 
    calibration     predicted probability versus observed rate. A model can
                    rank well and still be badly calibrated, which matters if
                    scores feed a downstream risk calculation rather than a
                    ranked queue.
 
MODEL REGISTRATION
------------------
The model is logged with an inferred signature and an input example. The
signature records expected column names and dtypes, so MLflow rejects a
malformed request at serving time rather than producing a silently wrong
prediction. This is the same train-serve skew defence as keeping preprocessing
inside the Pipeline, applied at the boundary.
"""
 
from __future__ import annotations
 
import json
import os
 
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
 
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
 
from aml_detection.baseline import rule_scores
from aml_detection.config import REPORTS_DIR
from aml_detection.features import FEATURE_COLUMNS, TARGET, feature_matrix
from aml_detection.modeling.train import BASELINE_PR_AUC, K_VALUES
from aml_detection.splitting import load_split
 
EXPERIMENT = "aml_detection"
REGISTERED_MODEL = "aml_detection_xgboost"
 
#: Recall the operating threshold targets. Reflects the cost asymmetry: a
#: missed STR is a regulatory failure, a false positive costs review time.
TARGET_RECALL = 0.60
 
 
# --------------------------------------------------------------------------
# Model retrieval
# --------------------------------------------------------------------------
 
def load_best_model(experiment: str = EXPERIMENT):
    """Fetch the highest val PR-AUC run from MLflow.
 
    Selection is by validation score, which is what validation is for. The test
    split plays no part in choosing which model is evaluated.
    """
    import mlflow
 
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty:
        raise RuntimeError(
            f"No runs in experiment '{experiment}'. "
            "Run: python -m aml_detection.modeling.train"
        )
 
    best = runs.sort_values("metrics.pr_auc", ascending=False).iloc[0]
    name = best.get("tags.mlflow.runName", "unknown")
    logger.info(
        f"Selected '{name}' (run {best.run_id[:8]}) "
        f"with validation PR-AUC {best['metrics.pr_auc']:.4f}"
    )
 
    model = mlflow.sklearn.load_model(f"runs:/{best.run_id}/model")
    return model, best.run_id, name
 
 
# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
 
def choose_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> dict:
    """Lowest score achieving the target recall on VALIDATION.
 
    Selected on validation and applied unchanged to test. Picking the threshold
    that looks best on test would fit the evaluation set and overstate
    precision -- the same error as tuning hyperparameters on test, in a smaller
    and easier-to-miss form.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
 
    # precision_recall_curve returns one more precision/recall point than
    # thresholds, so trim the trailing point before aligning.
    viable = np.where(recall[:-1] >= target_recall)[0]
    if len(viable) == 0:
        logger.warning(
            f"Target recall {target_recall:.0%} unreachable on validation. "
            "Falling back to the threshold with maximum recall."
        )
        index = int(np.argmax(recall[:-1]))
    else:
        # Highest threshold still meeting the target: best precision for that
        # recall level.
        index = int(viable[-1])
 
    return {
        "threshold": float(thresholds[index]),
        "val_precision": float(precision[index]),
        "val_recall": float(recall[index]),
        "target_recall": target_recall,
    }
 
 
# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
 
def headline_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Ranking quality, independent of any threshold."""
    pr_auc = average_precision_score(y_true, scores)
    base_rate = float(y_true.mean())
 
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": base_rate,
        "pr_auc_vs_baseline": float(pr_auc / BASELINE_PR_AUC),
        "pr_auc_vs_random": float(pr_auc / base_rate),
        "positives": int(y_true.sum()),
        "rows": int(len(y_true)),
    }
 
 
def operating_point_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Confusion counts at the chosen threshold.
 
    false_negatives is the number an AML function cares about most: each one is
    laundering that was not reported.
    """
    flagged = scores >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())
 
    return {
        "threshold": float(threshold),
        "alerts": int(flagged.sum()),
        "alert_rate": float(flagged.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "alerts_per_catch": (tp + fp) / tp if tp else np.nan,
    }
 
 
def recall_at_k_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Performance at fixed alert volumes.
 
    The operating question for an AML team: capacity is fixed, so of the k
    cases reviewable per period, how many are real and what share of laundering
    is caught?
    """
    base_rate = y_true.mean()
    rows = {}
 
    for k in K_VALUES:
        k = min(k, len(scores))
        top_k = np.argpartition(-scores, k - 1)[:k]
        caught = int(y_true[top_k].sum())
 
        rows[k] = {
            "caught": caught,
            "precision": caught / k,
            "recall": caught / y_true.sum() if y_true.sum() else np.nan,
            "lift": (caught / k) / base_rate if base_rate else np.nan,
        }
 
    return pd.DataFrame(rows).T.rename_axis("k")
 
 
def segment_recall(
    test: pd.DataFrame, scores: np.ndarray, threshold: float, column: str
) -> pd.DataFrame:
    """Recall broken down by a categorical column.
 
    Aggregate recall hides blind spots. The model draws over half its
    importance from one payment format, so it may catch almost nothing on the
    others -- which an investigator needs to know before trusting the queue.
    """
    frame = test.assign(_flagged=scores >= threshold, _score=scores)
 
    stats = frame.groupby(column, observed=True).apply(
        lambda g: pd.Series({
            "rows": len(g),
            "positives": int(g[TARGET].sum()),
            "caught": int((g["_flagged"] & (g[TARGET] == 1)).sum()),
            "alerts": int(g["_flagged"].sum()),
        }),
        include_groups=False,
    )
 
    stats["recall"] = stats["caught"] / stats["positives"].replace(0, np.nan)
    stats["precision"] = stats["caught"] / stats["alerts"].replace(0, np.nan)
    return stats.sort_values("positives", ascending=False)
 
 
def calibration_table(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed rate, by score decile.
 
    A model can rank correctly while being badly calibrated. Ranking is what
    matters for an alert queue, but calibration matters if scores feed a
    downstream risk figure. Note that class weighting distorts calibration by
    design -- scores will overstate probability, which is expected here.
    """
    frame = pd.DataFrame({"score": scores, "actual": y_true})
    frame["bin"] = pd.qcut(frame["score"], bins, duplicates="drop", labels=False)
 
    return (
        frame.groupby("bin")
        .agg(n=("actual", "size"),
             mean_predicted=("score", "mean"),
             observed_rate=("actual", "mean"))
        .assign(gap=lambda d: d["mean_predicted"] - d["observed_rate"])
    )
 
 
def baseline_comparison(test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Model against the rules baseline at identical alert volumes.
 
    The comparison an AML function would actually make: holding investigator
    capacity constant, how much more laundering does the model surface?
    """
    y_true = test[TARGET].to_numpy()
    rules = rule_scores(test).to_numpy()
 
    rows = {}
    for k in K_VALUES:
        k = min(k, len(scores))
 
        model_caught = int(y_true[np.argpartition(-scores, k - 1)[:k]].sum())
        # Rule counts are coarse integers with many ties; jitter breaks them
        # randomly rather than by row order, which would favour early rows.
        jitter = np.random.default_rng(42).random(len(rules)) * 0.5
        rules_caught = int(y_true[np.argpartition(-(rules + jitter), k - 1)[:k]].sum())
 
        rows[k] = {
            "model_caught": model_caught,
            "rules_caught": rules_caught,
            "model_precision": model_caught / k,
            "rules_precision": rules_caught / k,
            "improvement": model_caught / rules_caught if rules_caught else np.nan,
        }
 
    return pd.DataFrame(rows).T.rename_axis("k")
 
 
# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
 
def register_model(model, X_sample: pd.DataFrame, metrics: dict, run_name: str) -> None:
    """Log the evaluated model with a signature and register it.
 
    infer_signature records the expected input columns and dtypes plus the
    output shape. At serving time MLflow validates incoming requests against
    it, so a missing or misnamed column raises rather than producing a
    silently wrong prediction.
 
    The input example is stored alongside, giving anyone loading the model a
    concrete illustration of the expected payload without reading the code.
    """
    import mlflow
    from mlflow.models import infer_signature
 
    sample = X_sample.head(100)
    signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
 
    with mlflow.start_run(run_name=f"test_eval_{run_name}"):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_param("evaluated_model", run_name)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("split", "test")
 
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample,
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL,
        )
 
    logger.success(
        f"Registered '{REGISTERED_MODEL}' with signature "
        f"({len(signature.inputs.inputs)} inputs)"
    )
 
 
# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
 
def evaluate_on_test(register: bool = True) -> dict:
    """Run the full evaluation. Intended to be executed once."""
    logger.warning(
        "Evaluating on the TEST split. Every result below is an estimate of "
        "unseen performance ONLY because no decision was made against this "
        "data. Do not tune anything in response to these numbers."
    )
 
    model, run_id, run_name = load_best_model()
 
    val, test = load_split("val"), load_split("test")
    X_val, y_val = feature_matrix(val), val[TARGET].to_numpy()
    X_test, y_test = feature_matrix(test), test[TARGET].to_numpy()
 
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
 
    # Threshold from validation, applied unchanged to test.
    operating = choose_threshold(y_val, scores_val)
    logger.info(
        f"Threshold {operating['threshold']:.6f} chosen on validation "
        f"(recall {operating['val_recall']:.3f}, "
        f"precision {operating['val_precision']:.4f})"
    )
 
    headline = headline_metrics(y_test, scores_test)
    at_threshold = operating_point_metrics(y_test, scores_test, operating["threshold"])
 
    # A threshold that alerts on almost nothing usually means the validation
    # and test score distributions differ, not that the model is precise.
    if at_threshold["alerts"] < 10:
        logger.warning(
            f"Only {at_threshold['alerts']} alerts at the chosen threshold. "
            "The score distribution on test likely differs from validation -- "
            "check before interpreting precision at this operating point."
        )
 
    logger.success(
        f"TEST PR-AUC {headline['pr_auc']:.4f} "
        f"({headline['pr_auc_vs_baseline']:.1f}x rules baseline)"
    )
 
    # A large validation-to-test drop means the model does not hold up on
    # later data, which is the failure mode a temporal split exists to expose.
    import mlflow
    val_pr_auc = float(
        mlflow.get_run(run_id).data.metrics.get("pr_auc", np.nan)
    )
    drift = val_pr_auc - headline["pr_auc"]
    if not np.isnan(drift):
        logger.info(f"validation {val_pr_auc:.4f} -> test {headline['pr_auc']:.4f} "
                    f"({drift:+.4f})")
        if drift > 0.05:
            logger.warning(
                "Test performance is materially below validation. The model "
                "degrades on later data -- expected to some degree with "
                "evolving typologies, but worth reporting explicitly."
            )
 
    tables = {
        "recall_at_k": recall_at_k_table(y_test, scores_test),
        "by_payment_format": segment_recall(
            test, scores_test, operating["threshold"], "payment_format"
        ),
        "by_entity_type": segment_recall(
            test, scores_test, operating["threshold"], "src_entity_type"
        ),
        "calibration": calibration_table(y_test, scores_test),
        "vs_baseline": baseline_comparison(test, scores_test),
    }
    tables["by_payment_format"] = tables["by_payment_format"]
 
    blind = tables["by_payment_format"].query("positives >= 20 and recall < 0.2")
    if len(blind):
        logger.warning(
            f"Recall below 20% on: {blind.index.tolist()}. These channels are "
            "effectively unmonitored by this model."
        )
 
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORTS_DIR / f"test_{name}.csv")
 
    summary = {
        "model": run_name,
        "run_id": run_id,
        **headline,
        "brier_score": float(brier_score_loss(y_test, scores_test)),
        "operating_point": {**operating, **at_threshold},
        "val_to_test_drop": float(drift) if not np.isnan(drift) else None,
    }
    print("summary",summary)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(summary, indent=2))
 
    if register:
        register_model(model, X_test, headline, run_name)
 
    for name, table in tables.items():
        logger.info(f"\n--- {name} ---\n{table.round(4).to_string()}")
 
    return summary
 
 
if __name__ == "__main__":
    evaluate_on_test()