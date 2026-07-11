import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.reasoning_compression.modeling import (
    random_forest_feature_importance,
    select_model_params,
)


def test_select_fit_and_evaluate_random_forest_pipeline() -> None:
    X = pd.DataFrame({
        "numeric_feature": list(range(20)),
        "binary_feature": [0, 1] * 10,
        "category": ["a", "b", "c", "d"] * 5,
    })
    y = pd.Series([0, 1] * 10)
    groups = pd.Series(range(20))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["numeric_feature"]),
            ("bin", "passthrough", ["binary_feature"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["category"],
            ),
        ]
    )

    param_grid = {
        "n_estimators": [5],
        "min_samples_leaf": [1],
        "class_weight": [None],
        "random_state": [0],
        "n_jobs": [1],
    }
    selection_results = select_model_params(
        model_name="Random forest",
        model_class=RandomForestClassifier,
        param_grid=param_grid,
        preprocessor=preprocessor,
        X_train=X,
        y_train=y,
        groups_train=groups,
        max_tuning_groups=None,
    )

    assert len(selection_results) == 1
    assert selection_results.loc[0, "params"]["n_estimators"] == 5

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(**selection_results.loc[0, "params"]),
            ),
        ]
    )
    model.fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }
    assert set(metrics) == {"accuracy", "balanced_accuracy", "roc_auc"}

    importances = random_forest_feature_importance(model)
    assert set(importances.columns) == {"feature", "importance"}
    assert not importances.empty
