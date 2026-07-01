import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.reasoning_compression.modeling import (
    evaluate_classifier,
    fit_selected_model,
    make_preprocessor,
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

    preprocessor = make_preprocessor(
        numeric_columns=["numeric_feature"],
        binary_columns=["binary_feature"],
        categorical_columns=["category"],
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

    model = fit_selected_model(
        model_class=RandomForestClassifier,
        selected_params=selection_results.loc[0, "params"],
        preprocessor=preprocessor,
        X_train=X,
        y_train=y,
    )

    metrics = evaluate_classifier(model, X, y)
    assert set(metrics) == {"accuracy", "balanced_accuracy", "roc_auc"}

    importances = random_forest_feature_importance(model)
    assert set(importances.columns) == {"feature", "importance"}
    assert not importances.empty
