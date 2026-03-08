"""
trainer.py — K-Fold 학습 · Optuna 튜닝 · 모델 저장
════════════════════════════════════════════════════
CatBoost / LightGBM / XGBoost / RandomForest를 동일한 인터페이스로 다루며,
Optuna 자동 튜닝 → K-Fold 최종 학습 → OOF 예측 생성까지 담당합니다.

⚠️ 데이터 누수 방지: 각 K-Fold 내부에서 DataPreprocessor.fit(train)으로
   훈련 데이터의 통계량만 학습하고, transform(val)으로 검증 데이터에 적용합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from src.config import get_model_params, get_evaluation_config, get_paths, get_feature_config
from src.preprocessor import DataPreprocessor, apply_smote
from src.utils import get_logger, ensure_dir, Timer

logger = get_logger(__name__)


# ─── 결과 컨테이너 ─────────────────────────────────────────────
@dataclass
class TrainerResult:
    """학습 결과를 담는 데이터 클래스."""
    oof_preds: np.ndarray                     # Out-of-Fold 예측 (확률 또는 값)
    models: list = field(default_factory=list)  # 각 Fold 모델 리스트
    best_params: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  모델 객체 생성
# ═══════════════════════════════════════════════════════════════
def _build_catboost(params: dict, task_type: str, seed: int):
    """CatBoost 모델 객체를 생성합니다."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    if task_type == "regression":
        return CatBoostRegressor(**params, random_seed=seed)
    return CatBoostClassifier(**params, random_seed=seed)


def _build_lightgbm(params: dict, task_type: str, seed: int):
    """LightGBM 모델 객체를 생성합니다."""
    import lightgbm as lgb

    if task_type == "regression":
        return lgb.LGBMRegressor(**params, random_state=seed)
    return lgb.LGBMClassifier(**params, random_state=seed)


def _build_xgboost(params: dict, task_type: str, seed: int):
    """XGBoost 모델 객체를 생성합니다."""
    import xgboost as xgb

    if task_type == "regression":
        return xgb.XGBRegressor(**params, random_state=seed)
    return xgb.XGBClassifier(**params, random_state=seed, use_label_encoder=False)


def _build_randomforest(params: dict, task_type: str, seed: int):
    """RandomForest 모델 객체를 생성합니다."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if task_type == "regression":
        return RandomForestRegressor(**params, random_state=seed)
    return RandomForestClassifier(**params, random_state=seed)


def _build_balancedrandomforest(params: dict, task_type: str, seed: int):
    """BalancedRandomForest 모델 객체를 생성합니다. (Imbalanced-Learn)"""
    if task_type == "regression":
        raise ValueError("BalancedRandomForest는 분류(Classification) 태스크 전용입니다.")
    from imblearn.ensemble import BalancedRandomForestClassifier
    # 내부 서브샘플링 방식으로 불균형 해소
    return BalancedRandomForestClassifier(**params, random_state=seed)


def get_model(algorithm: str, params: dict, task_type: str = "classification", seed: int = 42):
    """알고리즘 문자열로 모델 객체를 반환합니다.
    
    Args:
        algorithm: "catboost" | "lightgbm" | "xgboost" | "randomforest"
        params: 모델 하이퍼파라미터 dict
        task_type: "classification" | "regression"
        seed: 랜덤 시드
    
    Returns:
        모델 객체 (sklearn-like API)
    """
    builders = {
        "catboost": _build_catboost,
        "lightgbm": _build_lightgbm,
        "xgboost": _build_xgboost,
        "randomforest": _build_randomforest,
        "balancedrandomforest": _build_balancedrandomforest,
    }
    if algorithm not in builders:
        raise ValueError(
            f"지원하지 않는 알고리즘: {algorithm} "
            f"(catboost, lightgbm, xgboost, randomforest, balancedrandomforest 중 선택)"
        )
    return builders[algorithm](params, task_type, seed)


# ═══════════════════════════════════════════════════════════════
#  K-Fold 헬퍼
# ═══════════════════════════════════════════════════════════════
def _get_kfold(
    task_type: str,
    n_splits: int,
    shuffle: bool,
    seed: int,
    split_strategy: str = "stratified",
):
    """지정된 전략(split_strategy)에 맞는 KFold 객체를 반환합니다."""
    # 시계열 분할
    if split_strategy == "timeseries":
        from sklearn.model_selection import TimeSeriesSplit
        return TimeSeriesSplit(n_splits=n_splits)

    # 그룹 분할
    if split_strategy == "group":
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=n_splits)

    # 랜덤 K-Fold
    if split_strategy == "kfold":
        from sklearn.model_selection import KFold
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # 기본값: StratifiedKFold (분류) or KFold (회귀)
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)


# ═══════════════════════════════════════════════════════════════
#  범주형 인코딩 헬퍼 (XGBoost · RandomForest 용)
# ═══════════════════════════════════════════════════════════════
def _encode_categoricals(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """범주형 컬럼을 LabelEncoding 합니다 (XGBoost·RF 용).
    
    Returns:
        (X_tr_encoded, X_val_encoded, encoders_dict)
    """
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    encoders = {}

    for col in cat_cols:
        if col not in X_tr.columns:
            continue
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        # val에 학습 시 없던 카테고리가 있으면 -1로 처리
        val_series = X_val[col].astype(str)
        X_val[col] = val_series.map(
            lambda x, _le=le: (
                _le.transform([x])[0] if x in _le.classes_ else -1
            )
        )
        encoders[col] = le

    return X_tr, X_val, encoders


# ═══════════════════════════════════════════════════════════════
#  Optuna 파라미터 제안 함수들
# ═══════════════════════════════════════════════════════════════
def _suggest_catboost_params(trial, optuna_cfg: dict) -> dict:
    """CatBoost용 Optuna 제안 파라미터를 생성합니다."""
    params_range = optuna_cfg.get("params", {})
    suggested = {}

    if "iterations" in params_range:
        r = params_range["iterations"]
        suggested["iterations"] = trial.suggest_int("iterations", r[0], r[1])
    if "depth" in params_range:
        r = params_range["depth"]
        suggested["depth"] = trial.suggest_int("depth", r[0], r[1])
    if "learning_rate" in params_range:
        r = params_range["learning_rate"]
        suggested["learning_rate"] = trial.suggest_float("learning_rate", r[0], r[1], log=True)
    if "l2_leaf_reg" in params_range:
        r = params_range["l2_leaf_reg"]
        suggested["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", r[0], r[1])
    if "bagging_temperature" in params_range:
        r = params_range["bagging_temperature"]
        suggested["bagging_temperature"] = trial.suggest_float("bagging_temperature", r[0], r[1])
    if "random_strength" in params_range:
        r = params_range["random_strength"]
        suggested["random_strength"] = trial.suggest_float("random_strength", r[0], r[1])
    if "border_count" in params_range:
        r = params_range["border_count"]
        suggested["border_count"] = trial.suggest_int("border_count", r[0], r[1])

    return suggested


def _suggest_lightgbm_params(trial, optuna_cfg: dict) -> dict:
    """LightGBM용 Optuna 제안 파라미터를 생성합니다."""
    params_range = optuna_cfg.get("params", {})
    suggested = {}

    if "n_estimators" in params_range:
        r = params_range["n_estimators"]
        suggested["n_estimators"] = trial.suggest_int("n_estimators", r[0], r[1])
    if "max_depth" in params_range:
        r = params_range["max_depth"]
        suggested["max_depth"] = trial.suggest_int("max_depth", r[0], r[1])
    if "learning_rate" in params_range:
        r = params_range["learning_rate"]
        suggested["learning_rate"] = trial.suggest_float("learning_rate", r[0], r[1], log=True)
    if "num_leaves" in params_range:
        r = params_range["num_leaves"]
        suggested["num_leaves"] = trial.suggest_int("num_leaves", r[0], r[1])
    if "min_child_samples" in params_range:
        r = params_range["min_child_samples"]
        suggested["min_child_samples"] = trial.suggest_int("min_child_samples", r[0], r[1])
    if "reg_alpha" in params_range:
        r = params_range["reg_alpha"]
        suggested["reg_alpha"] = trial.suggest_float("reg_alpha", r[0], r[1])
    if "reg_lambda" in params_range:
        r = params_range["reg_lambda"]
        suggested["reg_lambda"] = trial.suggest_float("reg_lambda", r[0], r[1])
    if "subsample" in params_range:
        r = params_range["subsample"]
        suggested["subsample"] = trial.suggest_float("subsample", r[0], r[1])
    if "colsample_bytree" in params_range:
        r = params_range["colsample_bytree"]
        suggested["colsample_bytree"] = trial.suggest_float("colsample_bytree", r[0], r[1])

    return suggested


def _suggest_xgboost_params(trial, optuna_cfg: dict) -> dict:
    """XGBoost용 Optuna 제안 파라미터를 생성합니다."""
    params_range = optuna_cfg.get("params", {})
    suggested = {}

    if "n_estimators" in params_range:
        r = params_range["n_estimators"]
        suggested["n_estimators"] = trial.suggest_int("n_estimators", r[0], r[1])
    if "max_depth" in params_range:
        r = params_range["max_depth"]
        suggested["max_depth"] = trial.suggest_int("max_depth", r[0], r[1])
    if "learning_rate" in params_range:
        r = params_range["learning_rate"]
        suggested["learning_rate"] = trial.suggest_float("learning_rate", r[0], r[1], log=True)
    if "min_child_weight" in params_range:
        r = params_range["min_child_weight"]
        suggested["min_child_weight"] = trial.suggest_int("min_child_weight", r[0], r[1])
    if "gamma" in params_range:
        r = params_range["gamma"]
        suggested["gamma"] = trial.suggest_float("gamma", r[0], r[1])
    if "subsample" in params_range:
        r = params_range["subsample"]
        suggested["subsample"] = trial.suggest_float("subsample", r[0], r[1])
    if "colsample_bytree" in params_range:
        r = params_range["colsample_bytree"]
        suggested["colsample_bytree"] = trial.suggest_float("colsample_bytree", r[0], r[1])
    if "reg_alpha" in params_range:
        r = params_range["reg_alpha"]
        suggested["reg_alpha"] = trial.suggest_float("reg_alpha", r[0], r[1])
    if "reg_lambda" in params_range:
        r = params_range["reg_lambda"]
        suggested["reg_lambda"] = trial.suggest_float("reg_lambda", r[0], r[1])

    return suggested


def _suggest_randomforest_params(trial, optuna_cfg: dict) -> dict:
    """RandomForest용 Optuna 제안 파라미터를 생성합니다."""
    params_range = optuna_cfg.get("params", {})
    suggested = {}

    if "n_estimators" in params_range:
        r = params_range["n_estimators"]
        suggested["n_estimators"] = trial.suggest_int("n_estimators", r[0], r[1])
    if "max_depth" in params_range:
        r = params_range["max_depth"]
        suggested["max_depth"] = trial.suggest_int("max_depth", r[0], r[1])
    if "min_samples_split" in params_range:
        r = params_range["min_samples_split"]
        suggested["min_samples_split"] = trial.suggest_int("min_samples_split", r[0], r[1])
    if "min_samples_leaf" in params_range:
        r = params_range["min_samples_leaf"]
        suggested["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", r[0], r[1])
    if "max_features" in params_range:
        choices = params_range["max_features"]
        # 문자열과 숫자가 섞여 있을 수 있음
        str_choices = [str(c) for c in choices]
        selected = trial.suggest_categorical("max_features", str_choices)
        # float 변환 시도
        try:
            suggested["max_features"] = float(selected)
        except ValueError:
            suggested["max_features"] = selected

    return suggested


def _suggest_balancedrandomforest_params(trial, optuna_cfg: dict) -> dict:
    """BalancedRandomForest용 Optuna 제안 파라미터를 생성합니다."""
    # RandomForest와 사실상 동일한 파라미터 공간 공유
    return _suggest_randomforest_params(trial, optuna_cfg)


# ─── suggest 함수 매핑 ─────────────────────────────────────────
_SUGGEST_FN_MAP = {
    "catboost": _suggest_catboost_params,
    "lightgbm": _suggest_lightgbm_params,
    "xgboost": _suggest_xgboost_params,
    "randomforest": _suggest_randomforest_params,
    "balancedrandomforest": _suggest_balancedrandomforest_params,
}


def _compute_fold_score(
    y_val: np.ndarray,
    val_pred: np.ndarray,
    val_proba: np.ndarray | None,
    metric: str,
    task_type: str,
) -> float:
    """단일 Fold의 지표를 계산합니다."""
    from sklearn.metrics import (
        log_loss, roc_auc_score, f1_score, accuracy_score,
        mean_squared_error, mean_absolute_error,
        precision_score, recall_score,
    )

    metric = metric.lower()

    if task_type == "regression":
        if metric == "rmse":
            return mean_squared_error(y_val, val_pred, squared=False)
        elif metric == "mae":
            return mean_absolute_error(y_val, val_pred)
        return mean_squared_error(y_val, val_pred, squared=False)

    # classification
    if metric == "logloss" and val_proba is not None:
        return log_loss(y_val, val_proba)
    elif metric == "auc" and val_proba is not None:
        if val_proba.ndim == 2 and val_proba.shape[1] == 2:
            return roc_auc_score(y_val, val_proba[:, 1])
        elif val_proba.ndim == 1:
            return roc_auc_score(y_val, val_proba)
        return roc_auc_score(y_val, val_proba, multi_class="ovr", average="macro")
    elif metric == "f1_macro":
        return f1_score(y_val, val_pred, average="macro")
    elif metric == "precision_macro":
        return precision_score(y_val, val_pred, average="macro")
    elif metric == "recall_macro":
        return recall_score(y_val, val_pred, average="macro")
    elif metric == "accuracy":
        return accuracy_score(y_val, val_pred)
    elif val_proba is not None:
        return log_loss(y_val, val_proba)
    return f1_score(y_val, val_pred, average="macro")


# ═══════════════════════════════════════════════════════════════
#  fit_kwargs 빌더
# ═══════════════════════════════════════════════════════════════
def _build_fit_kwargs(
    algorithm: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    early_stopping: int,
    verbose: int,
) -> dict[str, Any]:
    """알고리즘별 fit() 키워드 인자를 구성합니다."""
    fit_kwargs: dict[str, Any] = {}

    if algorithm == "catboost":
        fit_kwargs = {
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": early_stopping,
            "verbose": verbose,
            "cat_features": cat_cols if cat_cols else None,
        }
    elif algorithm == "lightgbm":
        import lightgbm as lgb
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.early_stopping(early_stopping, verbose=(verbose > 0)),
                lgb.log_evaluation(period=verbose if verbose > 0 else 0),
            ],
            "categorical_feature": cat_cols if cat_cols else "auto",
        }
    elif algorithm == "xgboost":
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": (verbose > 0),
        }
    # randomforest, balancedrandomforest: fit_kwargs stays empty (no eval_set / early_stopping)

    return fit_kwargs


# ═══════════════════════════════════════════════════════════════
#  Optuna 튜닝
# ═══════════════════════════════════════════════════════════════
def run_optuna_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> dict:
    """Optuna를 사용하여 최적 하이퍼파라미터를 탐색합니다.
    
    Args:
        X: 피처 DataFrame
        y: 타겟 Series
        config: 전체 설정 dict
    
    Returns:
        best_params dict
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_cfg = get_model_params(config)
    eval_cfg = get_evaluation_config(config)
    feat_cfg = get_feature_config(config)

    algorithm = model_cfg["algorithm"]
    n_splits = model_cfg["n_splits"]
    seed = model_cfg["seed"]
    task_type = model_cfg["task_type"]
    fixed_params = model_cfg["fixed_params"].copy()
    optuna_cfg = model_cfg["optuna"]
    early_stopping = model_cfg["early_stopping_rounds"]

    target_metric = eval_cfg.get("optuna_target_metric", "logloss")
    direction = eval_cfg.get("optuna_direction", "minimize")
    n_trials = optuna_cfg.get("n_trials", 50)
    timeout = optuna_cfg.get("timeout")

    split_strategy = model_cfg.get("split_strategy", "stratified")
    group_col = model_cfg.get("group_col", None)
    groups = X[group_col] if group_col and group_col in X.columns else None

    cat_cols = [c for c in feat_cfg["cat_cols"] if c in X.columns]
    needs_encoding = algorithm in ("xgboost", "randomforest", "balancedrandomforest")

    suggest_fn = _SUGGEST_FN_MAP[algorithm]

    def objective(trial):
        suggested = suggest_fn(trial, optuna_cfg)
        params = {**fixed_params, **suggested}

        kf = _get_kfold(task_type, n_splits, True, seed, split_strategy)
        scores = []

        # split 호출 시 groups 파라미터 전달 (GroupKFold 등의 경우 필요함)
        for train_idx, val_idx in kf.split(X, y, groups=groups):
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 결측치 처리 및 인코딩 (훈련 데이터 기준 fit → 검증에 transform)
            dp = DataPreprocessor(config)
            X_tr = dp.fit_transform(X_tr, y_tr)
            X_val = dp.transform(X_val)

            # 범주형 인코딩 (XGBoost · RandomForest)
            if needs_encoding and cat_cols:
                X_tr, X_val, _ = _encode_categoricals(X_tr, X_val, cat_cols)

            try:
                # SMOTE 적용 (훈련 데이터에만)
                if task_type == "classification":
                    X_tr, y_tr = apply_smote(X_tr, y_tr, config)

                model = get_model(algorithm, params, task_type, seed)

                fit_kwargs = _build_fit_kwargs(algorithm, X_val, y_val, cat_cols, early_stopping, 0)
                model.fit(X_tr, y_tr, **fit_kwargs)

                # 예측
                val_pred = model.predict(X_val)
                val_proba = None
                if task_type == "classification" and hasattr(model, "predict_proba"):
                    val_proba = model.predict_proba(X_val)

                score = _compute_fold_score(y_val, val_pred, val_proba, target_metric, task_type)
                scores.append(score)
            except Exception as e:
                import traceback
                import optuna
                logger.warning(f"[Optuna OOM/Error] Trial 중단 (건너뜀) - 이유: {e}")
                logger.debug(traceback.format_exc())
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    logger.info(f"Optuna 튜닝 시작 (algorithm={algorithm}, trials={n_trials}, metric={target_metric}, direction={direction})")
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Optuna 완료 — Best score: {study.best_value:.6f}")
    logger.info(f"Best params: {best}")
    return best


# ═══════════════════════════════════════════════════════════════
#  최종 K-Fold 학습
# ═══════════════════════════════════════════════════════════════
def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    best_params: dict | None = None,
) -> TrainerResult:
    """최종 하이퍼파라미터로 K-Fold 학습을 수행하고 OOF 예측·모델을 반환합니다.
    
    Args:
        X: 피처 DataFrame
        y: 타겟 Series
        config: 전체 설정 dict
        best_params: Optuna에서 얻은 최적 파라미터 (None이면 고정 파라미터만 사용)
    
    Returns:
        TrainerResult (oof_preds, models, best_params, feature_names)
    """
    model_cfg = get_model_params(config)
    feat_cfg = get_feature_config(config)
    paths = get_paths(config)

    algorithm = model_cfg["algorithm"]
    n_splits = model_cfg["n_splits"]
    shuffle = model_cfg["shuffle"]
    seed = model_cfg["seed"]
    task_type = model_cfg["task_type"]
    fixed_params = model_cfg["fixed_params"].copy()
    early_stopping = model_cfg["early_stopping_rounds"]

    split_strategy = model_cfg.get("split_strategy", "stratified")
    group_col = model_cfg.get("group_col", None)
    groups = X[group_col] if group_col and group_col in X.columns else None

    cat_cols = [c for c in feat_cfg["cat_cols"] if c in X.columns]
    needs_encoding = algorithm in ("xgboost", "randomforest", "balancedrandomforest")

    # 파라미터 병합
    params = {**fixed_params, **(best_params or {})}

    kf = _get_kfold(task_type, n_splits, shuffle, seed, split_strategy)

    # OOF 초기화
    if task_type == "classification":
        n_classes = y.nunique()
        if n_classes <= 2:
            oof_preds = np.zeros(len(X))
        else:
            oof_preds = np.zeros((len(X), n_classes))
    else:
        oof_preds = np.zeros(len(X))

    models = []

    logger.info(f"최종 K-Fold 학습 시작 (algorithm={algorithm}, n_splits={n_splits}, strategy={split_strategy})")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups), 1):
        logger.info(f"── Fold {fold_idx}/{n_splits} ──")

        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 결측치 처리 및 인코딩 (훈련 데이터 기준 fit → 검증에 transform)
        dp = DataPreprocessor(config)
        X_tr = dp.fit_transform(X_tr, y_tr)
        X_val = dp.transform(X_val)
        logger.info(f"  DataPreprocessor: 훈련 통계량으로 결측치 및 인코딩 처리 완료")

        # 범주형 인코딩
        if needs_encoding and cat_cols:
            X_tr, X_val, _ = _encode_categoricals(X_tr, X_val, cat_cols)

        # SMOTE 적용 (훈련 데이터에만)
        if task_type == "classification":
            X_tr, y_tr = apply_smote(X_tr, y_tr, config)

        model = get_model(algorithm, params, task_type, seed)

        # verbose 설정
        verbose = fixed_params.get("verbose", 100)
        if isinstance(verbose, bool):
            verbose = 100 if verbose else 0

        fit_kwargs = _build_fit_kwargs(algorithm, X_val, y_val, cat_cols, early_stopping, verbose)

        with Timer(f"Fold {fold_idx} 학습", logger):
            model.fit(X_tr, y_tr, **fit_kwargs)

        # OOF 예측
        if task_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if oof_preds.ndim == 1:
                oof_preds[val_idx] = proba[:, 1] if proba.ndim == 2 else proba
            else:
                oof_preds[val_idx] = proba
        else:
            oof_preds[val_idx] = model.predict(X_val)

        models.append(model)

        # 모델 저장
        model_dir = paths.get("model_dir", "models")
        ensure_dir(model_dir)

        if algorithm == "catboost":
            save_path = Path(model_dir) / f"{algorithm}_fold{fold_idx}.cbm"
            model.save_model(str(save_path))
        else:
            save_path = Path(model_dir) / f"{algorithm}_fold{fold_idx}.pkl"
            joblib.dump(model, save_path)
        logger.info(f"  모델 저장: {save_path}")

    logger.info("K-Fold 학습 완료")

    return TrainerResult(
        oof_preds=oof_preds,
        models=models,
        best_params=params,
        feature_names=X.columns.tolist(),
    )


# ═══════════════════════════════════════════════════════════════
#  전체 데이터 최종 학습 (Hold-out 평가용)
# ═══════════════════════════════════════════════════════════════
def train_full_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    best_params: dict | None = None,
) -> Any:
    """전체 훈련 데이터로 최종 단일 모델을 학습합니다.
    
    K-Fold 개발이 완료된 후, 확정된 하이퍼파라미터로
    전체 훈련 세트에 대해 모델 하나를 학습하고 저장합니다.
    
    Args:
        X_train: 전체 훈련 피처 DataFrame
        y_train: 전체 훈련 타겟 Series
        config: 전체 설정 dict
        best_params: 확정된 하이퍼파라미터 (None이면 고정 파라미터만 사용)
    
    Returns:
        학습된 최종 모델 객체
    """
    model_cfg = get_model_params(config)
    feat_cfg = get_feature_config(config)
    paths = get_paths(config)

    algorithm = model_cfg["algorithm"]
    seed = model_cfg["seed"]
    task_type = model_cfg["task_type"]
    fixed_params = model_cfg["fixed_params"].copy()

    cat_cols = [c for c in feat_cfg["cat_cols"] if c in X_train.columns]
    needs_encoding = algorithm in ("xgboost", "randomforest", "balancedrandomforest")

    # DataPreprocessor: 전체 훈련 데이터로 fit
    dp = DataPreprocessor(config)
    X_train = dp.fit_transform(X_train, y_train)

    # 범주형 인코딩
    encoders = {}
    if needs_encoding and cat_cols:
        from sklearn.preprocessing import LabelEncoder
        X_train = X_train.copy()
        for col in cat_cols:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                encoders[col] = le

    # 파라미터 병합
    params = {**fixed_params, **(best_params or {})}
    model = get_model(algorithm, params, task_type, seed)

    logger.info(f"최종 단일 모델 학습 시작 (algorithm={algorithm}, samples={len(X_train)})")

    with Timer("최종 모델 학습", logger):
        if algorithm == "catboost":
            model.fit(
                X_train, y_train,
                cat_features=cat_cols if cat_cols else None,
                verbose=fixed_params.get("verbose", 100),
            )
        elif algorithm == "lightgbm":
            import lightgbm as lgb
            model.fit(
                X_train, y_train,
                categorical_feature=cat_cols if cat_cols else "auto",
            )
        elif algorithm == "xgboost":
            model.fit(X_train, y_train, verbose=(fixed_params.get("verbosity", 0) > 0))
        else:
            model.fit(X_train, y_train)

    # 모델 저장
    model_dir = paths.get("model_dir", "models")
    ensure_dir(model_dir)

    if algorithm == "catboost":
        save_path = Path(model_dir) / f"{algorithm}_final.cbm"
        model.save_model(str(save_path))
    else:
        save_path = Path(model_dir) / f"{algorithm}_final.pkl"
        joblib.dump(model, save_path)
    logger.info(f"최종 모델 저장: {save_path}")

    # DataPreprocessor와 인코더도 저장 (홀드아웃 평가 시 필요)
    dp_path = Path(model_dir) / "data_preprocessor.pkl"
    joblib.dump(dp, dp_path)
    logger.info(f"DataPreprocessor 저장: {dp_path}")

    if encoders:
        enc_path = Path(model_dir) / "label_encoders.pkl"
        joblib.dump(encoders, enc_path)
        logger.info(f"LabelEncoders 저장: {enc_path}")

    return model, dp, encoders

