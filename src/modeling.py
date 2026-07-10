from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .feature_engineering import DroneFeatureEngineer
from .preprocessing import make_preprocessor

RANDOM_STATE = 42

def build_pipeline(model_name: str) -> Pipeline:
    model_name = model_name.lower()

    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        scale_numeric = True

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        scale_numeric = False

    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        scale_numeric = False

    else:
        raise ValueError("model_name must be one of: logistic_regression, random_forest, hist_gradient_boosting")

    return Pipeline(
        steps=[
            ("feature_engineering", DroneFeatureEngineer(drop_raw=True)),
            ("preprocessing", make_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", model),
        ]
    )


# Hyperparameter search spaces, one per model, keyed to the "classifier__*"
# step of the pipeline built by build_pipeline().
#
# NOTE: this function was imported by 02_Model_Training.ipynb
# (`from src.modeling import make_random_search`) but was missing from the
# uploaded module, so the notebook could not run end-to-end from a clean
# environment. The distributions below are a reasonable reconstruction that
# reproduces the parameter *names* seen in the notebook's best_params_ output
# (classifier__C / classifier__penalty for logistic regression,
# classifier__max_depth for random forest, classifier__l2_regularization for
# hist gradient boosting). The exact ranges were not recoverable from the
# notebook outputs, so double-check them against whatever you actually ran
# before treating re-generated numbers as identical to your original results.
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "classifier__C": loguniform(1e-3, 1e2),
        "classifier__penalty": ["l2"],
    },
    "random_forest": {
        "classifier__n_estimators": randint(200, 600),
        "classifier__max_depth": randint(3, 25),
        "classifier__min_samples_split": randint(2, 20),
        "classifier__min_samples_leaf": randint(1, 10),
        "classifier__max_features": ["sqrt", "log2", None],
    },
    "hist_gradient_boosting": {
        "classifier__learning_rate": loguniform(1e-2, 3e-1),
        "classifier__max_iter": randint(100, 400),
        "classifier__max_leaf_nodes": randint(15, 63),
        "classifier__max_depth": randint(3, 15),
        "classifier__l2_regularization": uniform(0, 1),
    },
}


def make_random_search(
    model_name: str,
    n_iter: int = 20,
    scoring: str = "roc_auc",
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Build an (unfitted) RandomizedSearchCV over build_pipeline(model_name).

    Uses stratified CV since the target is imbalanced (~60/40).
    """
    model_name = model_name.lower()
    if model_name not in PARAM_DISTRIBUTIONS:
        raise ValueError(
            f"model_name must be one of: {list(PARAM_DISTRIBUTIONS.keys())}"
        )

    pipeline = build_pipeline(model_name)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[model_name],
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
