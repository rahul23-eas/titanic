"""
Shared ML pipeline components for the Titanic classification project.

This module defines preprocessing and estimator factories used
consistently across training, evaluation, inference, FastAPI,
and Streamlit so that MLflow and joblib artifacts remain stable.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# FEATURE DEFINITIONS (MUST MATCH TRAINING & INFERENCE)
# =============================================================================

NUM_FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
]

CAT_FEATURES = [
    "sex",
]


# =============================================================================
# PREPROCESSING PIPELINES
# =============================================================================

def build_preprocessing() -> ColumnTransformer:
    """
    Build preprocessing pipeline for the Titanic dataset.
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_FEATURES),
            ("cat", categorical_pipeline, CAT_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessing


# =============================================================================
# ESTIMATOR FACTORY
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured classifier instance.
    """

    if name == "logistic":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )

    elif name == "ridge":
        return RidgeClassifier()

    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(
            random_state=42
        )

    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        )

    elif name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")


# =============================================================================
# OPTIONAL PIPELINE BUILDER (BASELINE / PCA)
# =============================================================================

def build_pipeline(model_name: str, use_pca: bool = False) -> Pipeline:
    """
    Build a full sklearn Pipeline including preprocessing,
    optional PCA, and a classifier.
    """

    preprocessing = build_preprocessing()
    model = make_estimator_for_name(model_name)

    steps = [("preprocessing", preprocessing)]

    if use_pca:
        steps.append(("pca", PCA(n_components=0.95)))

    steps.append(("model", model))

    return Pipeline(steps)
