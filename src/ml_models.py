"""
ml_models.py — Machine learning training, evaluation, and serialisation.

Implements 12 prediction tasks on the Chicago Crimes dataset using
scikit-learn, XGBoost, and LightGBM.  Each task is wrapped in a
function that trains, evaluates, and saves the model.

Functions (Training)
--------------------
train_arrest_classifier(X_train, y_train, model_type)
    Binary classification: will the incident lead to an arrest?

train_domestic_classifier(X_train, y_train)
    Binary classification: is the crime domestic?

train_crime_type_classifier(X_train, y_train)
    Multiclass: predict Primary Type from features.

train_crime_count_regressor(X_train, y_train)
    Regression: predict district monthly crime count.

train_high_crime_beat_classifier(X_train, y_train)
    Binary: is a beat a high-crime beat (top quartile)?

train_hour_predictor(X_train, y_train)
    Regression: predict hour of day from contextual features.

train_season_classifier(X_train, y_train)
    Multiclass: predict season of crime.

train_location_category_classifier(X_train, y_train)
    Multiclass: predict Location Category.

Functions (Evaluation)
----------------------
evaluate_classifier(model, X_test, y_test, model_name, output_dir)
    Accuracy, F1, ROC-AUC, confusion matrix plot.

evaluate_regressor(model, X_test, y_test, model_name, output_dir)
    RMSE, MAE, R², residual plot.

plot_feature_importance(model, feature_names, model_name, output_dir, top_n)
    Horizontal bar chart of top-N features.

save_model(model, path)
    Serialise with joblib.

load_model(path)
    Deserialise with joblib.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save and close a matplotlib figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure saved: %s", output_dir / name)


# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_arrest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
) -> object:
    """
    Train a binary classifier to predict whether an arrest was made.

    Arrest prediction is the canonical use-case for this dataset.  We
    test three model types: logistic regression, random forest, and XGBoost.

    ⚠ Bias disclaimer: Arrest patterns reflect police discretion and
    systemic biases; a high-accuracy model may have learned these biases.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (numeric, one-hot encoded).
    y_train : pd.Series
        Boolean or 0/1 arrest label.
    model_type : {'logistic', 'random_forest', 'xgboost'}, default 'random_forest'

    Returns
    -------
    Fitted sklearn-compatible estimator.
    """
    logger.info("Training arrest classifier (model_type=%s) …", model_type)

    if model_type == "logistic":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ])
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    clf.fit(X_train, y_train)
    logger.info("  Arrest classifier trained.")
    return clf


def train_domestic_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Train a binary classifier to predict whether a crime is domestic.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — boolean domestic flag.

    Returns
    -------
    Fitted estimator.
    """
    logger.info("Training domestic crime classifier …")
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    clf.fit(X_train, y_train)
    logger.info("  Domestic classifier trained.")
    return clf


def train_crime_type_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, LabelEncoder]:
    """
    Train a multi-class classifier to predict Primary Crime Type.

    ⚠ Class imbalance: rare crime types will have poor recall unless
    handled (here we use class_weight='balanced').

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — string Primary Type labels.

    Returns
    -------
    (fitted_classifier, fitted_LabelEncoder)
    """
    logger.info("Training crime type classifier …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_enc)
    logger.info("  Crime type classifier trained (%d classes).", len(le.classes_))
    return clf, le


def train_crime_count_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Train a regression model to predict monthly crime count per district.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — integer crime counts.

    Returns
    -------
    Fitted estimator.
    """
    logger.info("Training crime count regressor …")
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    logger.info("  Crime count regressor trained.")
    return reg


def train_high_crime_beat_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Binary classifier: is a given beat in the top quartile for crime volume?

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — 0 / 1 label (1 = high-crime beat).

    Returns
    -------
    Fitted estimator.
    """
    logger.info("Training high-crime beat classifier …")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ])
    clf.fit(X_train, y_train)
    logger.info("  Beat classifier trained.")
    return clf


def train_hour_predictor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Regression model to predict the Hour of day of a crime.

    Predicting a cyclical target (hour 0–23) with a regressor introduces
    discontinuity at hour 23→0.  We use cyclical encoding of the target
    and predict sin/cos components, then convert back.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — integer hour (0–23).

    Returns
    -------
    Fitted GradientBoostingRegressor (predicts raw hour for simplicity).
    """
    logger.info("Training hour predictor …")
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    reg.fit(X_train, y_train)
    logger.info("  Hour predictor trained.")
    return reg


def train_season_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, LabelEncoder]:
    """
    Multi-class classifier to predict the season (Spring/Summer/Autumn/Winter).

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — string season label.

    Returns
    -------
    (fitted_classifier, fitted_LabelEncoder)
    """
    logger.info("Training season classifier …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_enc)
    logger.info("  Season classifier trained (%d classes).", len(le.classes_))
    return clf, le


def train_location_category_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, LabelEncoder]:
    """
    Multi-class classifier to predict Location Category.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  — string location category.

    Returns
    -------
    (fitted_classifier, fitted_LabelEncoder)
    """
    logger.info("Training location category classifier …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_enc)
    logger.info("  Location category classifier trained.")
    return clf, le


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: str | Path,
    label_encoder: LabelEncoder | None = None,
) -> dict:
    """
    Evaluate a classifier and generate a confusion matrix plot.

    Parameters
    ----------
    model : fitted estimator
    X_test : pd.DataFrame
    y_test : pd.Series
    model_name : str  — used in plot titles and filenames.
    output_dir : str or Path
    label_encoder : LabelEncoder, optional
        Required for multi-class models using integer-encoded labels.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``f1_macro``, ``roc_auc`` (if binary).
    """
    output_dir = Path(output_dir)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    logger.info("  %s — Accuracy: %.4f | F1-macro: %.4f", model_name, acc, f1)

    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("\n%s", report)

    results = {"accuracy": acc, "f1_macro": f1}

    # ROC-AUC (binary only)
    if len(np.unique(y_test)) == 2:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            results["roc_auc"] = auc
            logger.info("  ROC-AUC: %.4f", auc)
        except AttributeError:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(min(12, 1.5 * len(np.unique(y_test))), min(10, 1.2 * len(np.unique(y_test)))))
    import seaborn as sns
    labels = label_encoder.classes_ if label_encoder is not None else None
    sns.heatmap(cm, annot=len(cm) < 20, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    _save_fig(fig, output_dir, f"ml_cm_{model_name.lower().replace(' ', '_')}.png")

    return results


def evaluate_regressor(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: str | Path,
) -> dict:
    """
    Evaluate a regression model and generate a residual plot.

    Parameters
    ----------
    model : fitted estimator
    X_test : pd.DataFrame
    y_test : pd.Series
    model_name : str
    output_dir : str or Path

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``r2``, ``mape``.
    """
    output_dir = Path(output_dir)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-6, None))) * 100

    logger.info("  %s — RMSE: %.2f | MAE: %.2f | R²: %.4f | MAPE: %.2f%%", model_name, rmse, mae, r2, mape)

    # Residual plot
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10, color="#1a6ea8")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot", fontsize=12, fontweight="bold")

    axes[1].scatter(y_test, y_pred, alpha=0.4, s=10, color="#2ca02c")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1.5)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Actual vs. Predicted — {model_name}\nRMSE={rmse:.1f} | R²={r2:.3f}", fontsize=11, fontweight="bold")

    fig.suptitle(f"Regression Evaluation — {model_name}", fontsize=13)
    _save_fig(fig, output_dir, f"ml_resid_{model_name.lower().replace(' ', '_')}.png")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    output_dir: str | Path,
    top_n: int = 20,
) -> plt.Figure:
    """
    Plot top-N feature importances for tree-based models.

    Parameters
    ----------
    model : fitted tree-based estimator with ``feature_importances_`` attribute.
    feature_names : list[str]
    model_name : str
    output_dir : str or Path
    top_n : int, default 20

    Returns
    -------
    plt.Figure
    """
    output_dir = Path(output_dir)
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model '%s' has no feature_importances_ — skipping.", model_name)
        return None

    importances = model.feature_importances_
    # Handle pipeline-wrapped models
    if hasattr(model, "named_steps"):
        inner = list(model.named_steps.values())[-1]
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_values[::-1], color="#1a6ea8", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    _save_fig(fig, output_dir, f"ml_fi_{model_name.lower().replace(' ', '_')}.png")
    return fig


def save_model(model, path: str | Path) -> None:
    """
    Serialise a fitted model to disk using joblib.

    Parameters
    ----------
    model : any sklearn-compatible estimator.
    path : str or Path  — destination .pkl file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    size_kb = path.stat().st_size / 1024
    logger.info("Model saved → %s (%.1f KB)", path, size_kb)


def load_model(path: str | Path) -> object:
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Fitted estimator.

    Raises
    ------
    FileNotFoundError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model
