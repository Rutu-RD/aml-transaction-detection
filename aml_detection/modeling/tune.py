from __future__ import annotations
 
import json
import os
 
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
 
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
 
from aml_detection.config import MODELS_DIR, PROJ_ROOT, REPORTS_DIR
from aml_detection.features import FEATURE_COLUMNS, TARGET, feature_matrix
from aml_detection.modeling.train import (
    ALL_FEATURES,
    BASELINE_PR_AUC,
    RANDOM_STATE,
    build_preprocessor,
    evaluate,
    leakage_check,
    ModelSpec,
)
from aml_detection.splitting import load_split
 
PARAMS_FILE = PROJ_ROOT / "params.yaml"
 
 
def load_tuning_params() -> dict:
    """Read the tuning configuration from params.yaml.
 
    Keeping search grids in params.yaml rather than in code makes them a
    tracked input: `dvc repro` re-runs tuning when a grid changes, and
    `dvc params diff` shows exactly what moved between experiments. Grids
    hardcoded in a module are invisible to both.
    """
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"{PARAMS_FILE} not found. It must contain a 'tuning' block with "
            "n_splits, scoring, and a 'grids' mapping per model."
        )
 
    config = yaml.safe_load(PARAMS_FILE.read_text())
    if "tuning" not in config:
        raise KeyError(f"No 'tuning' block in {PARAMS_FILE}")
 
    tuning = config["tuning"]
    for key in ("n_splits", "scoring", "grids"):
        if key not in tuning:
            raise KeyError(f"params.yaml tuning block is missing '{key}'")
    return tuning
 
 
# --------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------
 
def build_search_specs(
    scale_pos_weight: float,
    grids: dict | None = None,
) -> list[tuple[ModelSpec, dict]]:
    """Model specifications paired with the grid to search over each.
 
    Grids come from params.yaml. Parameter names there are prefixed with
    model__ because the estimator sits inside a Pipeline whose first step is
    the preprocessor -- sklearn resolves nested parameters by that path.
 
    A model with no entry in params.yaml is skipped rather than silently
    searched with defaults, so the config file is the single source of truth
    for what gets tuned.
    """
    from sklearn.linear_model import LogisticRegression
 
    grids = load_tuning_params()["grids"] if grids is None else grids
    specs: list[tuple[ModelSpec, dict]] = []
 
    if "logreg" in grids:
        specs.append(
            (
                ModelSpec(
                    name="logreg",
                    estimator=LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
                grids["logreg"],
            )
        )
 
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed -- tuning the primary candidate is skipped")
        return specs
 
    if "xgboost" in grids:
        specs.append(
            (
                ModelSpec(
                    name="xgboost",
                    estimator=XGBClassifier(
                        n_estimators=400,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="aucpr",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    handles_nan=True,
                ),
                grids["xgboost"],
            )
        )
 
    if not specs:
        raise ValueError(
            "No models to tune. Check the 'grids' block in params.yaml."
        )
    return specs
 
 
# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
 
def tune_one(
    spec: ModelSpec,
    grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_splits: int,
    scoring: str,
) -> tuple[Pipeline, dict, pd.DataFrame, dict]:
    """Grid search one model with time-ordered folds, then score on validation.
 
    Two different numbers come out of this and they answer different questions:
 
        cv_pr_auc   mean PR-AUC across the time-ordered folds inside TRAIN.
                    Used to choose hyperparameters.
 
        val_pr_auc  PR-AUC on the held-out validation split, which the search
                    never saw. Used to compare tuned against untuned, and
                    against other models.
 
    Selecting on cv and reporting val keeps the comparison honest: if the two
    diverge sharply, the grid has been overfitted to the fold structure.
    """
    n_combinations = int(np.prod([len(v) for v in grid.values()]))
    logger.info(
        f"Tuning {spec.name}: {n_combinations} combinations x {n_splits} folds "
        f"= {n_combinations * n_splits} fits"
    )
 
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(handles_nan=spec.handles_nan)),
        ("model", spec.estimator),
    ])
 
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=TimeSeriesSplit(n_splits=n_splits),   # NOT KFold: order matters
        n_jobs=1,          # the estimators already parallelise internally
        refit=True,        # refit the winner on the full training split
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
 
    best = search.best_estimator_
    scores = best.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val.to_numpy(), scores)
 
    metrics["cv_pr_auc"] = float(search.best_score_)
    metrics["cv_pr_auc_std"] = float(
        search.cv_results_["std_test_score"][search.best_index_]
    )
    # A large gap between fold score and held-out score means the grid has been
    # fitted to the fold structure rather than to generalisable signal.
    metrics["cv_val_gap"] = metrics["cv_pr_auc"] - metrics["pr_auc"]
 
    logger.success(
        f"{spec.name}: cv PR-AUC {metrics['cv_pr_auc']:.4f} "
        f"(+/-{metrics['cv_pr_auc_std']:.4f}) | "
        f"val PR-AUC {metrics['pr_auc']:.4f} "
        f"({metrics['pr_auc_vs_baseline']:.1f}x baseline)"
    )
    if abs(metrics["cv_val_gap"]) > 0.05:
        logger.warning(
            f"cv and validation scores differ by {metrics['cv_val_gap']:+.4f}. "
            "Either the folds are not representative of the validation period, "
            "or the grid has been overfitted."
        )
 
    results = pd.DataFrame(search.cv_results_)
    return best, metrics, results, search.best_params_
 
 
# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
 
def run_tuning(
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    use_mlflow: bool = True,
) -> pd.DataFrame:
    """Tune every model and return a comparison table.
 
    Runs are logged as cv_<model_name> so they appear alongside the untuned
    runs from train.py in the same experiment.
    """
    params = load_tuning_params()
    n_splits = params["n_splits"]
    scoring = params["scoring"]
    grids = params["grids"]
 
    train = load_split("train") if train is None else train
    val = load_split("val") if val is None else val
 
    X_train, y_train = train[list(ALL_FEATURES)], train[TARGET]
    X_val, y_val = val[list(ALL_FEATURES)], val[TARGET]
 
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(
        f"train {len(X_train):,} rows, {int(y_train.sum()):,} positives | "
        f"{n_splits} time-ordered folds | scoring={scoring} | "
        f"models={list(grids)}"
    )
 
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("aml_detection")
        except ImportError:
            logger.warning("mlflow not installed -- running without tracking")
            use_mlflow = False
 
    results: dict[str, dict] = {}
 
    for spec, grid in build_search_specs(scale_pos_weight, grids):
        run_name = f"cv_{spec.name}"
 
        if use_mlflow:
            import mlflow
            with mlflow.start_run(run_name=run_name):
                best, metrics, cv_results, best_params = tune_one(
                    spec, grid, X_train, y_train, X_val, y_val, n_splits, scoring
                )
 
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in best_params.items()})
                mlflow.log_param("cv_folds", n_splits)
                mlflow.log_param("cv_strategy", "TimeSeriesSplit")
                mlflow.log_param("scoring", scoring)
                mlflow.log_param("n_candidates", len(cv_results))
                mlflow.log_param("n_features", len(FEATURE_COLUMNS))
 
                mlflow.log_metrics(metrics)
 
                importance = leakage_check(best, spec)
                if len(importance):
                    importance.head(20).to_csv(
                        REPORTS_DIR / f"importance_{run_name}.csv"
                    )
                    mlflow.log_metric("top_feature_share", float(importance.iloc[0]))
 
                # The full grid is logged as an artifact so the search can be
                # inspected later without rerunning it.
                cv_path = REPORTS_DIR / f"cv_results_{spec.name}.csv"
                cv_results.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))
 
                mlflow.sklearn.log_model(
                    best, name="model", serialization_format="cloudpickle"
                )
        else:
            best, metrics, cv_results, best_params = tune_one(
                spec, grid, X_train, y_train, X_val, y_val, n_splits, scoring
            )
            leakage_check(best, spec)
            cv_results.to_csv(REPORTS_DIR / f"cv_results_{spec.name}.csv", index=False)
 
        metrics["best_params"] = json.dumps(
            {k.replace("model__", ""): v for k, v in best_params.items()}
        )
        results[run_name] = metrics
 
    comparison = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
 
    best_name = comparison.index[0]
    best_score = comparison.iloc[0]["pr_auc"]
    logger.success(f"Best tuned: {best_name} with val PR-AUC {best_score:.4f}")
 
    if best_score < BASELINE_PR_AUC:
        logger.error(
            f"No tuned model beat the rules baseline ({BASELINE_PR_AUC:.4f})."
        )
 
    return comparison
 
 
def main() -> None:
    comparison = run_tuning()
 
    comparison.to_csv(REPORTS_DIR / "tuning_comparison.csv")
 
    headline = comparison.iloc[0]
    (REPORTS_DIR / "tuning_metrics.json").write_text(
        json.dumps(
            {
                "best_model": comparison.index[0],
                "best_params": headline["best_params"],
                "cv_pr_auc": float(headline["cv_pr_auc"]),
                "val_pr_auc": float(headline["pr_auc"]),
                "pr_auc_vs_baseline": float(headline["pr_auc_vs_baseline"]),
                "recall_at_1000": float(headline["recall_at_1000"]),
            },
            indent=2,
        )
    )
 
    display_columns = [
        "cv_pr_auc", "cv_pr_auc_std", "pr_auc", "cv_val_gap",
        "pr_auc_vs_baseline", "recall_at_1000", "best_params",
    ]
    logger.info("\n" + comparison[display_columns].round(4).to_string())
 
 
if __name__ == "__main__":
    main()