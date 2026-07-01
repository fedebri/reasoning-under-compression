"""Shared modeling utilities for reasoning-compression notebooks."""

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_SELECTION_METRIC = "roc_auc"
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_VALIDATION_RANDOM_STATE = 43
DEFAULT_MAX_TUNING_GROUPS = 30_000

RANDOM_FOREST_PARAM_GRID: dict[str, list[object]] = {
    "n_estimators": [150, 300],
    "min_samples_leaf": [2, 5],
    "class_weight": ["balanced"],
    "random_state": [42],
    "n_jobs": [-1],
}

GRADIENT_BOOSTING_PARAM_GRID: dict[str, list[object]] = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [150],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 0.1],
    "class_weight": [None, "balanced"],
    "random_state": [42],
}

__all__ = [
    "DEFAULT_SELECTION_METRIC",
    "DEFAULT_VALIDATION_SIZE",
    "DEFAULT_VALIDATION_RANDOM_STATE",
    "DEFAULT_MAX_TUNING_GROUPS",
    "RANDOM_FOREST_PARAM_GRID",
    "GRADIENT_BOOSTING_PARAM_GRID",
    "evaluate_classifier",
    "make_preprocessor",
    "make_pipeline",
    "make_grouped_tuning_data",
    "select_model_params",
    "fit_selected_model",
    "classification_report_frame",
    "confusion_matrix_frame",
    "random_forest_feature_importance",
]


class ProbabilisticClassifier(Protocol):
    """Classifier protocol required by the notebook evaluation helpers."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


def evaluate_classifier(
    model: ProbabilisticClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict[str, float]:
    """Evaluate binary classifier performance on a fixed evaluation split."""
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_eval, pred),
        "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
    }

    if y_eval.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_eval, proba)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def make_preprocessor(
    numeric_columns: list[str],
    binary_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for model-ready feature tables."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            ("bin", "passthrough", binary_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )


def make_pipeline(
    model_class: type[BaseEstimator],
    model_params: Mapping[str, object],
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Build a fresh preprocessing/model pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", model_class(**dict(model_params))),
        ]
    )


def make_grouped_tuning_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    max_groups: int | None,
    random_state: int = DEFAULT_VALIDATION_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Optionally subsample training groups for faster validation tuning."""
    if max_groups is None:
        return X_train, y_train, groups_train

    unique_groups = pd.Series(groups_train.drop_duplicates().to_numpy())
    if len(unique_groups) <= max_groups:
        return X_train, y_train, groups_train

    sampled_groups = set(
        unique_groups.sample(n=max_groups, random_state=random_state).to_numpy()
    )
    tuning_mask = groups_train.isin(sampled_groups)

    return (
        X_train.loc[tuning_mask],
        y_train.loc[tuning_mask],
        groups_train.loc[tuning_mask],
    )


def select_model_params(
    model_name: str,
    model_class: type[BaseEstimator],
    param_grid: Mapping[str, Sequence[object]],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    max_tuning_groups: int | None = DEFAULT_MAX_TUNING_GROUPS,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    """Rank parameter settings using one grouped validation split.

    The split is drawn from the training data only. The held-out test split is
    untouched by model selection.
    """
    X_tune, y_tune, groups_tune = make_grouped_tuning_data(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        max_groups=max_tuning_groups,
        random_state=DEFAULT_VALIDATION_RANDOM_STATE,
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=DEFAULT_VALIDATION_SIZE,
        random_state=DEFAULT_VALIDATION_RANDOM_STATE,
    )
    fit_idx, val_idx = next(splitter.split(X_tune, y_tune, groups=groups_tune))

    X_fit = X_tune.iloc[fit_idx]
    X_val = X_tune.iloc[val_idx]
    y_fit = y_tune.iloc[fit_idx]
    y_val = y_tune.iloc[val_idx]

    rows = []
    for params in ParameterGrid(param_grid):
        model = make_pipeline(model_class, params, preprocessor)
        model.fit(X_fit, y_fit)
        metrics = evaluate_classifier(model, X_val, y_val)
        rows.append({
            "model": model_name,
            "params": params,
            "n_tuning_rows": len(X_tune),
            "n_fit_rows": len(X_fit),
            "n_validation_rows": len(X_val),
            **metrics,
        })

    if not rows:
        raise ValueError("Parameter grid produced no candidate models.")

    results = pd.DataFrame(rows)
    if selection_metric not in results.columns:
        raise KeyError(f"Selection metric not found in results: {selection_metric}")

    return (
        results
        .sort_values(
            [selection_metric, "balanced_accuracy"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def fit_selected_model(
    model_class: type[BaseEstimator],
    selected_params: Mapping[str, object],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the selected model configuration on the full training split."""
    model = make_pipeline(model_class, selected_params, preprocessor)
    model.fit(X_train, y_train)
    return model


def classification_report_frame(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a sklearn classification report as a tidy dataframe."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"model": model_name, "label": label, **values})
        else:
            rows.append({
                "model": model_name,
                "label": label,
                "precision": np.nan,
                "recall": np.nan,
                "f1-score": values,
                "support": len(y_true),
            })

    return pd.DataFrame(rows)


def confusion_matrix_frame(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a two-class confusion matrix as a labeled dataframe."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index(["actual_0", "actual_1"], name="actual"),
        columns=pd.Index(["pred_0", "pred_1"], name=model_name),
    )


def random_forest_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return feature importances from a fitted random-forest pipeline."""
    forest = model.named_steps["model"]
    if not isinstance(forest, RandomForestClassifier):
        raise TypeError(
            "random_forest_feature_importance expects a pipeline whose "
            "'model' step is a RandomForestClassifier."
        )

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = forest.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
