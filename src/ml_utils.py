"""
ml_utils.py
===========
Machine-learning model training, evaluation, and visualisation helpers for
the Chicago crime analysis project.

Provides:
- Feature matrix preparation and label encoding.
- Stratified k-fold cross-validation wrappers.
- Standardised evaluation reporters for classifiers and regressors.
- Confusion matrix, ROC-AUC, and feature-importance visualisation.
- SMOTE oversampling for imbalanced classification tasks.

Typical usage
-------------
>>> from src.ml_utils import prepare_features, evaluate_classifier, plot_feature_importance
>>> X, y = prepare_features(df, target="Arrest", cat_cols=["Primary Type", "Season"])
>>> evaluate_classifier(X, y, model_name="Random Forest")
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target: str,
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a feature matrix X and target series y from the crime DataFrame.

    Categorical columns are label-encoded (integer-coded). This is appropriate
    for tree-based models. For linear models, consider one-hot encoding instead.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered crime DataFrame.
    target : str
        Name of the column to predict.
    numeric_cols : list of str, optional
        Numeric feature columns to include. If None, all numeric columns
        except *target* are used.
    cat_cols : list of str, optional
        Categorical feature columns to label-encode and include.
    drop_cols : list of str, optional
        Columns to explicitly exclude from the feature matrix (e.g. ID
        columns or the raw date).

    Returns
    -------
    (X, y) : tuple
        X — pd.DataFrame of features.
        y — pd.Series of the target variable.

    Notes
    -----
    - Rows with NaN in *target* are dropped before returning.
    - Label encoding assigns an arbitrary integer to each category level.
      For tree models (Random Forest, XGBoost) this is acceptable; for
      linear models use ``pd.get_dummies`` instead.
    """
    df = df.copy()
    df = df.dropna(subset=[target])

    feature_cols: List[str] = []

    # Numeric features
    if numeric_cols:
        feature_cols += numeric_cols
    else:
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        num_candidates = [c for c in num_candidates if c != target]
        if drop_cols:
            num_candidates = [c for c in num_candidates if c not in drop_cols]
        feature_cols += num_candidates

    # Categorical features (label-encoded)
    if cat_cols:
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                feature_cols.append(col + "_enc")

    # Remove explicit drops and target from feature set
    if drop_cols:
        feature_cols = [c for c in feature_cols if c not in drop_cols]
    feature_cols = [c for c in feature_cols if c != target]

    X = df[feature_cols].copy()
    y = df[target].copy()

    log.info("Feature matrix: %d rows × %d features. Target: '%s'.", len(X), X.shape[1], target)
    return X, y


# ── Classification helpers ────────────────────────────────────────────────────

def evaluate_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Random Forest",
    n_splits: int = 5,
    use_smote: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate a classifier with stratified k-fold cross-validation.

    Supported model names (case-insensitive):
    - ``"Random Forest"``       → RandomForestClassifier
    - ``"Logistic Regression"`` → LogisticRegression
    - ``"Gradient Boosting"``   → GradientBoostingClassifier
    - ``"XGBoost"``             → XGBClassifier
    - ``"kNN"``                 → KNeighborsClassifier

    .. warning::
       SMOTE is applied within each fold's training data only — never on the
       test fold — to prevent data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix as returned by :func:`prepare_features`.
    y : pd.Series
        Target series (binary or multi-class).
    model_name : str
        Classifier to use. See above for valid values. Default ``"Random Forest"``.
    n_splits : int
        Number of stratified folds. Default 5.
    use_smote : bool
        Whether to apply SMOTE oversampling on the training fold to handle
        class imbalance. Default True.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mean and std of ``accuracy``, ``f1_macro``, ``roc_auc``
        across all folds.
    """
    model_map = {
        "random forest":       RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "logistic regression": LogisticRegression(max_iter=500, random_state=random_state),
        "gradient boosting":   GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "xgboost":             xgb.XGBClassifier(n_estimators=100, random_state=random_state,
                                                  eval_metric="logloss", verbosity=0),
        "knn":                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    clf = model_map.get(model_name.lower())
    if clf is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map.keys())}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s, aucs = [], [], []

    X_arr = X.fillna(0).values
    y_arr = pd.Categorical(y).codes  # encode labels to integers

    is_binary = len(np.unique(y_arr)) == 2

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # SMOTE oversampling (training fold only)
        if use_smote:
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=5)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            except Exception:
                pass  # SMOTE fails for tiny minority classes — proceed without

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))

        if y_prob is not None:
            try:
                auc = roc_auc_score(y_te, y_prob if not is_binary else y_prob[:, 1],
                                    multi_class="ovr" if not is_binary else "raise")
                aucs.append(auc)
            except Exception:
                pass

    results = {
        "model":        model_name,
        "accuracy_mean": round(np.mean(accs), 4),
        "accuracy_std":  round(np.std(accs),  4),
        "f1_macro_mean": round(np.mean(f1s),  4),
        "f1_macro_std":  round(np.std(f1s),   4),
        "roc_auc_mean":  round(np.mean(aucs), 4) if aucs else None,
        "roc_auc_std":   round(np.std(aucs),  4) if aucs else None,
    }

    log.info(
        "%s — Accuracy: %.4f ± %.4f | F1: %.4f ± %.4f | AUC: %s",
        model_name,
        results["accuracy_mean"], results["accuracy_std"],
        results["f1_macro_mean"], results["f1_macro_std"],
        f"{results['roc_auc_mean']:.4f}" if results["roc_auc_mean"] else "N/A",
    )
    return results


# ── Regression helpers ────────────────────────────────────────────────────────

def evaluate_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate a regression model with k-fold cross-validation.

    Supported model names:
    - ``"Linear Regression"``
    - ``"Ridge"``
    - ``"Random Forest"``
    - ``"XGBoost"``

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Continuous target (e.g. daily crime count).
    model_name : str
        Regressor to use. Default ``"XGBoost"``.
    n_splits : int
        Number of folds. Default 5.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Mean and std of ``mae``, ``rmse``, ``r2`` across folds.
    """
    model_map = {
        "linear regression": LinearRegression(),
        "ridge":             Ridge(alpha=1.0),
        "random forest":     RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "xgboost":           xgb.XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0),
    }

    reg = model_map.get(model_name.lower())
    if reg is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map.keys())}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_arr = X.fillna(0).values
    y_arr = y.values

    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in kf.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        r2s.append(r2_score(y_te, preds))

    results = {
        "model":     model_name,
        "mae_mean":  round(np.mean(maes),  2),
        "mae_std":   round(np.std(maes),   2),
        "rmse_mean": round(np.mean(rmses), 2),
        "rmse_std":  round(np.std(rmses),  2),
        "r2_mean":   round(np.mean(r2s),   4),
        "r2_std":    round(np.std(r2s),    4),
    }
    log.info(
        "%s — MAE: %.2f ± %.2f | RMSE: %.2f ± %.2f | R²: %.4f ± %.4f",
        model_name,
        results["mae_mean"],  results["mae_std"],
        results["rmse_mean"], results["rmse_std"],
        results["r2_mean"],   results["r2_std"],
    )
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a labelled, colour-coded confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list of str, optional
        Class name labels for the axes. If None, integer codes are used.
    title : str
        Plot title.
    figsize : tuple
        Figure dimensions.
    save_path : str, optional
        Save path for the figure; if None, the plot is displayed.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importances",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a horizontal bar chart of the top-N most important features.

    Works with any sklearn-compatible model that exposes
    ``feature_importances_`` (Random Forest, XGBoost, Gradient Boosting).

    Parameters
    ----------
    model
        A fitted model with a ``feature_importances_`` attribute.
    feature_names : list of str
        Names corresponding to model input features.
    top_n : int
        Number of top features to display. Default 20.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure here instead of displaying.

    Raises
    ------
    AttributeError
        If the model does not have ``feature_importances_``.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model of type {type(model).__name__} does not expose feature_importances_."
        )

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    importances.plot.barh(ax=ax, color="#2C7BB6", edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
