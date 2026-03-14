"""
evaluator.py — 평가 지표 계산 · 시각화 · 리포트 생성
═══════════════════════════════════════════════════════
OOF 예측을 기반으로 포괄적인 지표를 계산하고,
Confusion Matrix, ROC Curve, PR Curve, Feature Importance,
SHAP Summary / Waterfall / Dependence 등의 시각화를 저장합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 비-GUI 백엔드 (서버 환경 호환)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    brier_score_loss,
    cohen_kappa_score,
)

from src.config import get_evaluation_config, get_visualization_config, get_paths
from src.utils import get_logger, ensure_dir, format_number

logger = get_logger(__name__)

# 전역 플롯 스타일
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
})


# ═══════════════════════════════════════════════════════════════
#  1. 지표 계산
# ═══════════════════════════════════════════════════════════════
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    task_type: str = "classification",
) -> dict[str, float]:
    """포괄적인 평가 지표를 계산하여 dict로 반환합니다.
    
    분류 지표: Accuracy, Precision(Macro/Micro), Recall(Macro/Micro),
              F1(Macro/Micro/Weighted), ROC-AUC, LogLoss
    회귀 지표: RMSE, MAE, R²
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨 (분류) 또는 예측 값 (회귀)
        y_proba: 예측 확률 (분류 전용, None이면 확률 기반 지표 스킵)
        task_type: "classification" | "regression"
    
    Returns:
        { "metric_name": value, ... }
    """
    metrics: dict[str, float] = {}

    if task_type == "regression":
        metrics["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        # ── 라벨 기반 지표 ──
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # ── 확률 기반 지표 ──
        if y_proba is not None:
            try:
                metrics["logloss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass
            try:
                if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
                    proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                    metrics["pr_auc"] = float(average_precision_score(y_true, proba_pos))
                    metrics["brier_score"] = float(brier_score_loss(y_true, proba_pos))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    )
            except Exception:
                pass

    return metrics


# ═══════════════════════════════════════════════════════════════
#  2. 최적 임계값 탐색 (이진 분류)
# ═══════════════════════════════════════════════════════════════
def find_best_threshold(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    metric: str = "f1_macro",
    search_range: tuple[float, float] = (0.1, 0.9),
    search_step: float = 0.01,
) -> tuple[float, float]:
    """이진 분류에서 최적 임계값을 그리드 서치로 탐색합니다.
    
    Args:
        y_true: 실제 라벨
        y_proba_positive: positive 클래스 예측 확률
        metric: 최적화할 지표
        search_range: 탐색 범위 [start, end]
        search_step: 탐색 간격
    
    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.arange(search_range[0], search_range[1] + search_step, search_step)
    best_score = -1.0
    best_threshold = 0.5

    metric_fn = {
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
        "f1_micro": lambda yt, yp: f1_score(yt, yp, average="micro"),
        "precision_macro": lambda yt, yp: precision_score(yt, yp, average="macro"),
        "recall_macro": lambda yt, yp: recall_score(yt, yp, average="macro"),
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    }
    fn = metric_fn.get(metric, metric_fn["f1_macro"])

    for t in thresholds:
        preds = (y_proba_positive >= t).astype(int)
        score = fn(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    logger.info(
        f"최적 임계값: {format_number(best_threshold)} "
        f"(metric={metric}, score={format_number(best_score)})"
    )
    return best_threshold, best_score


# ═══════════════════════════════════════════════════════════════
#  3. 시각화 — 기본 차트
# ═══════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """Confusion Matrix를 시각화하여 저장합니다."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Confusion Matrix 저장: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """ROC Curve를 시각화하여 저장합니다 (이진 분류)."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
    auc_val = roc_auc_score(y_true, y_proba_positive)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"ROC Curve 저장: {save_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """Precision-Recall Curve를 시각화하여 저장합니다 (이진 분류)."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba_positive)
    ap = average_precision_score(y_true, y_proba_positive)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="navy", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"PR Curve 저장: {save_path}")


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """다중 분류용 OvR(One-vs-Rest) ROC Curve를 시각화합니다.

    각 클래스별 ROC 곡선과 Macro 평균 ROC 곡선을 하나의 그래프에 그립니다.

    Args:
        y_true: 실제 라벨 (정수 인코딩된 클래스)
        y_proba: 각 클래스별 예측 확률 (shape: n_samples x n_classes)
        save_path: 저장 경로
        dpi: 이미지 해상도
    """
    from sklearn.preprocessing import label_binarize
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    # 이진화 (OvR 방식)
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))  # type: ignore

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.5, color=color, alpha=0.8,
                label=f"Class {cls} (AUC = {auc_val:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro avg (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Multi-class ROC Curve (OvR, {n_classes} classes)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Multi-class ROC Curve 저장: {save_path}")


def plot_fold_scores(
    fold_scores: list[float],
    save_path: str | Path,
    metric_name: str = "score",
    dpi: int = 150,
) -> None:
    """K-Fold 각 Fold의 성능 점수를 시각화합니다.

    박스플롯 + 각 Fold 점수 산점도 + 평균선을 함께 그립니다.

    Args:
        fold_scores: Fold별 성능 점수 리스트
        save_path: 저장 경로
        metric_name: 지표 이름 (y축 레이블 및 제목에 표시)
        dpi: 이미지 해상도
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    scores = np.array(fold_scores)
    n_folds = len(scores)
    mean_score = scores.mean()
    std_score = scores.std()

    fig, ax = plt.subplots(figsize=(8, 5))

    # 박스플롯
    bp = ax.boxplot(scores, patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor="steelblue", alpha=0.4),
                    medianprops=dict(color="navy", linewidth=2))

    # 각 Fold 점수 산점도 (jitter)
    jitter = np.random.uniform(-0.1, 0.1, n_folds)
    ax.scatter(np.ones(n_folds) + jitter, scores,
               color="steelblue", s=60, zorder=5, alpha=0.8)

    # Fold 번호 레이블
    for i, score in enumerate(scores):
        ax.annotate(f"F{i+1}={score:.4f}", (1 + jitter[i], score),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color="dimgray")

    # 평균선
    ax.axhline(mean_score, color="crimson", linestyle="--", lw=1.5,
               label=f"Mean = {mean_score:.4f} ± {std_score:.4f}")

    ax.set_xticks([1])
    ax.set_xticklabels([f"{n_folds}-Fold CV"])
    ax.set_ylabel(metric_name)
    ax.set_title(f"K-Fold 성능 분포 ({metric_name})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Fold 성능 분포 저장: {save_path}")


def plot_feature_importance(
    model,
    feature_names: list[str],
    save_path: str | Path,
    top_n: int = 20,
    dpi: int = 150,
) -> None:
    """모델의 피처 중요도를 수평 바 차트로 시각화합니다."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    # 피처 중요도 추출
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    elif hasattr(model, "coef_"):
        # Logistic Regression 등 선형 모델: |계수| 절대값 사용
        coef = model.coef_
        if coef.ndim == 2:
            importances = np.abs(coef).mean(axis=0)  # 다중분류: 클래스별 평균
        else:
            importances = np.abs(coef)
    else:
        logger.warning("해당 모델은 feature_importances_ / coef_ 속성이 없어 중요도 시각화를 건너뜁니다.")
        return

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Feature Importance 저장: {save_path}")


def plot_calibration_curve_custom(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
    dpi: int = 150,
) -> None:
    """Calibration Curve (신뢰도 교정 곡선) 시각화"""
    from sklearn.calibration import calibration_curve
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    prob_true, prob_pred = calibration_curve(y_true, y_proba_positive, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="s", color="darkred", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Calibration Curve 저장: {save_path}")


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    threshold: float,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """예측 확률 산점도 (정답/오답 및 임계값 기준) 시각화"""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    preds = (y_proba_positive >= threshold).astype(int)
    correct = (preds == y_true)

    # DataFrame 구성 (stripplot 용)
    df_plot = pd.DataFrame({
        "True Label": y_true,
        "Predicted Probability": y_proba_positive,
        "Correct": correct
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(
        data=df_plot, x="True Label", y="Predicted Probability",
        hue="Correct", palette={True: "mediumseagreen", False: "crimson"},
        jitter=0.3, alpha=0.6, size=5, ax=ax
    )
    ax.axhline(threshold, color="navy", linestyle="--", lw=2, label=f"Threshold ({threshold:.2f})")
    ax.set_title(f"Prediction Probability Scatter (Threshold = {threshold:.2f})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Prediction Scatter 저장: {save_path}")


def plot_lift_gain_curve(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """Cumulative Gain Curve / Lift Curve 시각화 (간이 버전)"""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    # 확률 순 정렬
    df = pd.DataFrame({'y_true': y_true, 'proba': y_proba_positive})
    df = df.sort_values(by='proba', ascending=False).reset_index(drop=True)
    
    total_positives = df['y_true'].sum()
    if total_positives == 0:
        return
    
    df['cumulative_positives'] = df['y_true'].cumsum()
    df['percent_data'] = (np.arange(len(df)) + 1) / len(df)
    df['gain'] = df['cumulative_positives'] / total_positives
    
    # Lift = Gain / Percent Data
    df['lift'] = df['gain'] / df['percent_data']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cumulative Gains Curve
    ax1.plot(df['percent_data'], df['gain'], color='darkorange', lw=2, label='Model')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Baseline')
    ax1.set_xlabel('Percentage of sample (%)')
    ax1.set_ylabel('Gain')
    ax1.set_title('Cumulative Gain Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Lift Curve
    ax2.plot(df['percent_data'], df['lift'], color='teal', lw=2, label='Model')
    ax2.axhline(1.0, color='gray', lw=1, linestyle='--', label='Baseline')
    ax2.set_xlabel('Percentage of sample (%)')
    ax2.set_ylabel('Lift')
    ax2.set_title('Lift Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Lift/Gain Curve 저장: {save_path}")


# ═══════════════════════════════════════════════════════════════
#  4. 시각화 — SHAP
# ═══════════════════════════════════════════════════════════════
def _get_shap_explainer_and_values(model, X: pd.DataFrame):
    """SHAP Explainer와 shap_values를 생성합니다. (내부 헬퍼)
    
    TreeExplainer 호환 모델이 아니면 None을 반환합니다.
    (KernelExplainer는 너무 느려서 기본 적용하지 않음)
    """
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values
    except Exception as e:
        logger.warning(
            f"SHAP TreeExplainer 미지원 모델 (건너뜀): {type(model).__name__} — {e}"
        )
        return None, None


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    save_path: str | Path,
    max_display: int = 20,
    dpi: int = 150,
) -> None:
    """SHAP Summary Plot (Beeswarm)을 생성하여 저장합니다."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    try:
        import shap
        _, shap_values = _get_shap_explainer_and_values(model, X)
        if shap_values is None:
            return

        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP Summary 저장: {save_path}")
    except Exception as e:
        logger.warning(f"SHAP Summary 시각화 실패 (건너뜀): {e}")


def plot_shap_waterfall(
    model,
    X: pd.DataFrame,
    save_path: str | Path,
    sample_index: int = 0,
    max_display: int = 20,
    dpi: int = 150,
) -> None:
    """SHAP Waterfall Plot을 생성하여 저장합니다.
    
    단일 샘플에 대한 피처별 기여도를 폭포수 형태로 시각화합니다.
    
    Args:
        model: 학습된 모델
        X: 피처 DataFrame
        save_path: 저장 경로
        sample_index: 시각화할 샘플 인덱스
        max_display: 표시할 최대 피처 수
        dpi: 이미지 해상도
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X)

        # 다중 클래스인 경우 positive 클래스(인덱스 1)를 사용
        if isinstance(shap_explanation.values, list):
            explanation_slice = shap_explanation[sample_index, :, 1]
        elif shap_explanation.values.ndim == 3:
            explanation_slice = shap_explanation[sample_index, :, 1]
        else:
            explanation_slice = shap_explanation[sample_index]

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation_slice, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP Waterfall 저장: {save_path}")
    except Exception as e:
        logger.warning(f"SHAP Waterfall 시각화 실패 (건너뜀): {e}")


def plot_shap_dependence(
    model,
    X: pd.DataFrame,
    save_dir: str | Path,
    top_n: int = 5,
    dpi: int = 150,
) -> None:
    """상위 N개 피처에 대한 SHAP Dependence Plot을 생성합니다.
    
    각 피처가 예측에 미치는 영향을 산점도로 시각화합니다.
    자동으로 interaction feature도 color로 표시합니다.
    
    Args:
        model: 학습된 모델
        X: 피처 DataFrame
        save_dir: 저장 디렉토리 경로
        top_n: 시각화할 상위 피처 수
        dpi: 이미지 해상도
    """
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    try:
        import shap
        _, shap_values = _get_shap_explainer_and_values(model, X)

        # 다중 클래스면 positive 클래스 사용
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        if sv is None:
            return

        # 피처 중요도 순으로 상위 N개 선택
        mean_abs_shap = np.abs(sv).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_features = [X.columns[i] for i in top_indices]

        for feat in top_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feat, sv, X, ax=ax, show=False)
            fig.tight_layout()
            safe_name = feat.replace("/", "_").replace("\\", "_")
            fig.savefig(save_dir / f"shap_dependence_{safe_name}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"SHAP Dependence Plot {top_n}개 저장: {save_dir}")
    except Exception as e:
        logger.warning(f"SHAP Dependence 시각화 실패 (건너뜀): {e}")


# ═══════════════════════════════════════════════════════════════
#  5. 시각화 — 기타 XAI (PDP 등)
# ═══════════════════════════════════════════════════════════════
def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    save_dir: str | Path,
    feature_names: list[str],
    top_n: int = 5,
    dpi: int = 150,
) -> None:
    """상위 N개 피처에 대한 Partial Dependence Plot을 생성합니다."""
    from sklearn.inspection import PartialDependenceDisplay
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    try:
        # 피처 중요도(또는 SHAP 등)가 없으면 앞에서 N개 사용
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_features = feature_names[:top_n]

        fig, ax = plt.subplots(figsize=(12, min(8, top_n * 2)))
        
        # sklearn 1.2+ 
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X,
            features=top_features,
            grid_resolution=30,
            ax=ax,
            n_cols=min(3, len(top_features)),
        )
        fig.suptitle("Partial Dependence Plot", y=1.02, fontsize=16)
        fig.tight_layout()
        
        save_path = save_dir / "partial_dependence_plot.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"PDP 저장: {save_path}")
    except Exception as e:
        logger.warning(f"Partial Dependence Plot 시각화 실패 (건너뜀): {e}")


# ═══════════════════════════════════════════════════════════════
#  6. 리포트 생성
# ═══════════════════════════════════════════════════════════════
def generate_report(
    metrics: dict[str, float],
    config: dict,
    save_path: str | Path,
    best_params: dict | None = None,
    best_threshold: float | None = None,
) -> None:
    """평가 지표 요약을 텍스트 파일로 저장합니다."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    project_name = config.get("project", {}).get("name", "unknown")
    algorithm = config.get("model", {}).get("algorithm", "unknown")

    lines = [
        "=" * 60,
        f"  📊 평가 리포트 — {project_name}",
        f"  알고리즘: {algorithm}",
        "=" * 60,
        "",
    ]
    
    if best_threshold is not None:
        lines.append(f"  [적용된 최적 임계값]: {format_number(best_threshold)}\n")

    if best_params:
        lines.append("  [학습에 사용된 하이퍼파라미터]")
        for k, v in best_params.items():
            lines.append(f"    - {k}: {v}")
        lines.append("")

    lines.append("  [평가 지표]")
    for name, value in metrics.items():
        lines.append(f"    {name:<18s}: {format_number(value)}")

    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    save_path.write_text(report_text, encoding="utf-8")
    logger.info(f"평가 리포트 저장: {save_path}")

    # 콘솔에도 출력
    print(report_text)


# ═══════════════════════════════════════════════════════════════
#  6. 통합 평가 & 시각화 함수
# ═══════════════════════════════════════════════════════════════
def evaluate_and_visualize(
    y_true: np.ndarray,
    oof_preds: np.ndarray,
    models: list,
    feature_names: list[str],
    X: pd.DataFrame,
    config: dict,
    best_params: dict | None = None,
    best_threshold: float | None = None,
) -> dict[str, float]:
    """평가 지표 계산 + 시각화 + 리포트 생성을 일괄 처리합니다.
    
    Args:
        y_true: 실제 라벨
        oof_preds: OOF 예측 (확률 또는 값)
        models: 학습된 모델 리스트 (Fold별)
        feature_names: 피처 이름 리스트
        X: 피처 DataFrame (SHAP용)
        config: 전체 설정 dict
        best_params: 사용된 하이퍼파라미터
        best_threshold: 분류 결정을 위해 적용된 임계값
    
    Returns:
        계산된 지표 dict
    """
    eval_cfg = get_evaluation_config(config)
    vis_cfg = get_visualization_config(config)
    paths = get_paths(config)
    task_type = config.get("project", {}).get("task_type", "classification")

    figure_dir = Path(paths.get("figure_dir", "results/figures"))
    result_dir = Path(paths.get("result_dir", "results"))
    dpi = vis_cfg.get("dpi", 150)

    # ── 예측 라벨 결정 ──
    if task_type == "classification":
        threshold_cfg = eval_cfg.get("threshold_tuning", {})
        is_binary = oof_preds.ndim == 1

        if is_binary and threshold_cfg.get("enabled", False):
            best_t, _ = find_best_threshold(
                y_true,
                oof_preds,
                metric=threshold_cfg.get("metric", "f1_macro"),
                search_range=tuple(threshold_cfg.get("search_range", [0.1, 0.9])),
                search_step=threshold_cfg.get("search_step", 0.01),
            )
            y_pred = (oof_preds >= best_t).astype(int)
        elif is_binary:
            y_pred = (oof_preds >= 0.5).astype(int)
        else:
            y_pred = np.argmax(oof_preds, axis=1)

        y_proba = oof_preds
    else:
        y_pred = oof_preds
        y_proba = None

    # ── 지표 계산 ──
    metrics = calculate_metrics(y_true, y_pred, y_proba, task_type)
    logger.info(f"평가 지표: {metrics}")

    # ── 시각화: Confusion Matrix ──
    if task_type == "classification" and vis_cfg.get("confusion_matrix", True):
        plot_confusion_matrix(y_true, y_pred, figure_dir / "confusion_matrix.png", dpi)

    # ── 시각화: ROC Curve ──
    is_binary = (y_proba is not None) and (y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2))
    if task_type == "classification" and is_binary and vis_cfg.get("roc_curve", True):
        proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        plot_roc_curve(y_true, proba_pos, figure_dir / "roc_curve.png", dpi)

    # ── 시각화: 다중 분류 ROC Curve (OvR) ──
    is_multiclass = (y_proba is not None) and (y_proba.ndim == 2 and y_proba.shape[1] > 2)
    if task_type == "classification" and is_multiclass and vis_cfg.get("roc_curve", True):
        plot_multiclass_roc(y_true, y_proba, figure_dir / "roc_curve_multiclass.png", dpi)

    # ── 시각화: PR Curve ──
    if task_type == "classification" and is_binary and vis_cfg.get("pr_curve", True):
        proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        plot_pr_curve(y_true, proba_pos, figure_dir / "pr_curve.png", dpi)

    # ── 시각화: 신규 분류 평가지표 (Calibration, Scatter, Lift/Gain) ──
    if task_type == "classification" and is_binary:
        proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        threshold_to_plot = best_threshold if best_threshold is not None else 0.5
        
        if vis_cfg.get("calibration_curve", True):
            plot_calibration_curve_custom(y_true, proba_pos, figure_dir / "calibration_curve.png", dpi=dpi)
        
        if vis_cfg.get("prediction_scatter", True):
            plot_prediction_scatter(y_true, proba_pos, threshold_to_plot, figure_dir / "prediction_scatter.png", dpi=dpi)
        
        if vis_cfg.get("lift_gain_curve", True):
            plot_lift_gain_curve(y_true, proba_pos, figure_dir / "lift_gain_curve.png", dpi=dpi)

    # ── 시각화: Feature Importance ──
    if vis_cfg.get("feature_importance", True) and models:
        plot_feature_importance(
            models[0], feature_names, figure_dir / "feature_importance.png",
            top_n=vis_cfg.get("shap_max_display", 20), dpi=dpi,
        )

    # ── 시각화: SHAP Summary ──
    if vis_cfg.get("shap_summary", True) and models:
        # SHAP은 샘플이 너무 많으면 느리므로 최대 5000개로 제한
        X_shap = X.sample(n=min(5000, len(X)), random_state=42) if len(X) > 5000 else X
        plot_shap_summary(
            models[0], X_shap, figure_dir / "shap_summary.png",
            max_display=vis_cfg.get("shap_max_display", 20), dpi=dpi,
        )

    # ── 시각화: SHAP Waterfall ──
    if vis_cfg.get("shap_waterfall", False) and models:
        X_shap = X.sample(n=min(5000, len(X)), random_state=42) if len(X) > 5000 else X
        plot_shap_waterfall(
            models[0], X_shap, figure_dir / "shap_waterfall.png",
            sample_index=0,
            max_display=vis_cfg.get("shap_max_display", 20), dpi=dpi,
        )

    # ── 시각화: SHAP Dependence ──
    if vis_cfg.get("shap_dependence", False) and models:
        X_shap = X.sample(n=min(5000, len(X)), random_state=42) if len(X) > 5000 else X
        plot_shap_dependence(
            models[0], X_shap, figure_dir / "shap_dependence",
            top_n=vis_cfg.get("shap_dependence_features", 5), dpi=dpi,
        )

    # ── 시각화: PDP (Partial Dependence Plot) ──
    if vis_cfg.get("partial_dependence", True) and models:
        plot_partial_dependence(
            models[0], X, figure_dir, feature_names,
            top_n=vis_cfg.get("pdp_features", 5), dpi=dpi,
        )

    # ── 리포트 ──
    generate_report(metrics, config, result_dir / "evaluation_report.txt", best_params, best_threshold)

    # ── Classification Report (분류 전용) ──
    if task_type == "classification":
        report = classification_report(y_true, y_pred, zero_division=0)
        report_path = result_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"Classification Report 저장: {report_path}")

    return metrics
