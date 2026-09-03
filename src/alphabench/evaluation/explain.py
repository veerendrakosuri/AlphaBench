from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; called from the CLI, not a notebook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:  # noqa: N803 -- X/y convention
    """SHAP values for a tree model's positive-class probability, via TreeExplainer's
    fast exact path for LightGBM/XGBoost. Returns an (n_samples, n_features) array."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older SHAP API returns [class0, class1] for binary clf
        sv = sv[1]
    return np.asarray(sv)


def plot_global_importance(
    shap_values: np.ndarray, feature_names: list[str], path: Path, top_n: int = 20
) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(np.array(feature_names)[order][::-1], mean_abs[order][::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance — LightGBM h=1 (walk-forward final model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,  # noqa: N803 -- X/y convention
    path: Path,
    max_display: int = 20,
) -> None:
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_n_features(shap_values: np.ndarray, feature_names: list[str], n: int = 10) -> list[str]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [feature_names[i] for i in order]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def fold_feature_stability(fold_top_features: dict[int, list[str]]) -> dict:
    """Pairwise Jaccard overlap of each fold's own model's top-10 SHAP features.
    Features whose importance swings wildly between folds are noise-fitted, not signal
    (PROPOSAL section 7.3)."""
    years = sorted(fold_top_features)
    pairs = list(itertools.combinations(years, 2))
    overlaps = {
        f"{a}_vs_{b}": jaccard(fold_top_features[a], fold_top_features[b]) for a, b in pairs
    }
    mean_overlap = float(np.mean(list(overlaps.values()))) if overlaps else None
    return {
        "top_features_by_fold": fold_top_features,
        "pairwise_jaccard": overlaps,
        "mean_jaccard": mean_overlap,
    }
