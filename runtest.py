import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

from titanic_pipeline import build_preprocessing, make_estimator_for_name


# ------------------------
# Load data
# ------------------------
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.drop(columns=["Survived"])
y_train = train_df["Survived"]

X_test = test_df.drop(columns=["Survived"])
y_test = test_df["Survived"]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = [
    "logistic",
    "ridge",
    "histgradientboosting",
    "xgboost",
    "lightgbm",
]

pca_options = [False, True]

# ------------------------
# Run baseline experiments
# ------------------------
for model_name in models:
    for use_pca in pca_options:

        if model_name == "logistic" and use_pca:
            continue  # optional, but fine to skip

        with mlflow.start_run(run_name=f"{model_name}_baseline_pca_{use_pca}"):

            preprocessing = build_preprocessing()
            steps = [("preprocessing", preprocessing)]

            if use_pca:
                steps.append(("pca", PCA(n_components=0.95, random_state=42)))

            steps.append(("model", make_estimator_for_name(model_name)))

            pipeline = Pipeline(steps)

            cv_f1 = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="f1"
            ).mean()

            pipeline.fit(X_train, y_train)
            test_f1 = f1_score(y_test, pipeline.predict(X_test))

            mlflow.log_param("model_family", model_name)
            mlflow.log_param("uses_pca", use_pca)
            mlflow.log_param("is_tuned", False)
            mlflow.log_param("cv_folds", 3)

            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metric("test_f1", test_f1)

            mlflow.sklearn.log_model(pipeline, "model")
