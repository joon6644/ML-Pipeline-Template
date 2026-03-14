"""
src/config_validator.py — YAML 설정 유효성 검증
═══════════════════════════════════════════════
파이프라인 시작 전 YAML 설정 파일의 오타 및 잘못된 값을 조기에 감지합니다.
pydantic 없이 순수 Python으로 구현되어 추가 의존성 없이 동작합니다.

사용 예시 (config.py의 load_config 내에서 자동 호출됨)::

    from src.config_validator import validate_config
    validate_config(config)   # 오류 시 ValueError 발생
"""

from __future__ import annotations

from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)


# ── 허용 값 목록 ─────────────────────────────────────────────────
_VALID_TASK_TYPES = {"classification", "regression"}
_VALID_ALGORITHMS = {
    "catboost", "lightgbm", "xgboost", "randomforest", "balancedrandomforest",
    "easyensemble", "stacking", "logistic", "decisiontree", "svm", "knn", "naivebayes",
}
_VALID_SPLIT_STRATEGIES = {"stratified", "kfold", "group", "timeseries"}
_VALID_NUMERIC_MISSING = {"median", "mean", "zero", "knn", "iterative", "drop"}
_VALID_CATEGORICAL_MISSING = {"mode", "constant", "drop"}
_VALID_SCALING_METHODS = {"none", "standard", "robust", "minmax"}
_VALID_ENCODING_METHODS = {"none", "ordinal", "label", "target"}
_VALID_IMBALANCE_METHODS = {
    "none", "smote", "adasyn", "borderline_smote", "random_over",
    "random_under", "tomek", "enn", "smote_tomek", "smote_enn",
}
_VALID_OPTUNA_METRICS = {
    "logloss", "auc", "accuracy", "rmse", "mae", "r2", "mape",
    "f1_macro", "f1_micro", "f1_weighted",
    "f2_macro", "f2_micro", "f2_weighted",
    "f0.5_macro", "f0.5_micro", "f0.5_weighted",
}
_VALID_OPTUNA_DIRECTIONS = {"minimize", "maximize"}


class ConfigValidationError(ValueError):
    """YAML 설정 유효성 검증 실패 시 발생하는 예외."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            "\n\u001b[91m[설정 오류]\u001b[0m YAML 설정 파일에 다음 문제가 있습니다:\n"
            + "\n".join(f"  ❌ {e}" for e in errors)
        )


def validate_config(config: dict[str, Any]) -> None:
    """YAML 설정 전체를 검증하고 오류가 있으면 ConfigValidationError를 발생시킵니다.

    Args:
        config: load_config()로 로드한 설정 dict

    Raises:
        ConfigValidationError: 하나 이상의 설정 오류 발견 시
    """
    errors: list[str] = []

    _validate_project(config.get("project", {}), errors)
    _validate_model(config.get("model", {}), errors)
    _validate_features(config.get("features", {}), errors)
    _validate_evaluation(config.get("evaluation", {}), errors)
    _validate_paths(config.get("paths", {}), errors)

    if errors:
        raise ConfigValidationError(errors)

    logger.debug(f"설정 검증 완료 — 문제 없음 ({len(errors)}개 오류)")


def _validate_project(project: dict, errors: list) -> None:
    task_type = project.get("task_type", "")
    if task_type not in _VALID_TASK_TYPES:
        errors.append(
            f"project.task_type = '{task_type}' — "
            f"허용 값: {sorted(_VALID_TASK_TYPES)}"
        )

    seed = project.get("seed", 42)
    if not isinstance(seed, int) or seed < 0:
        errors.append(f"project.seed = {seed!r} — 0 이상의 정수여야 합니다.")

    mlflow_uri = project.get("mlflow_tracking_uri", "file:./mlruns")
    if not isinstance(mlflow_uri, str) or not mlflow_uri:
        errors.append("project.mlflow_tracking_uri — 비어 있는 문자열은 허용되지 않습니다.")


def _validate_model(model: dict, errors: list) -> None:
    algorithm = model.get("algorithm", "")
    if algorithm not in _VALID_ALGORITHMS:
        errors.append(
            f"model.algorithm = '{algorithm}' — "
            f"허용 값: {sorted(_VALID_ALGORITHMS)}"
        )

    n_splits = model.get("n_splits", 5)
    if not isinstance(n_splits, int) or n_splits < 2:
        errors.append(f"model.n_splits = {n_splits!r} — 2 이상의 정수여야 합니다.")

    split_strategy = model.get("split_strategy", "")
    if split_strategy and split_strategy not in _VALID_SPLIT_STRATEGIES:
        errors.append(
            f"model.split_strategy = '{split_strategy}' — "
            f"허용 값: {sorted(_VALID_SPLIT_STRATEGIES)}"
        )

    if split_strategy == "group" and not model.get("group_col"):
        errors.append(
            "model.split_strategy = 'group' 이지만 "
            "model.group_col 이 설정되지 않았습니다."
        )

    early_stop = model.get("early_stopping_rounds", 0)
    if not isinstance(early_stop, int) or early_stop < 0:
        errors.append(f"model.early_stopping_rounds = {early_stop!r} — 0 이상의 정수여야 합니다.")

    # Optuna
    optuna = model.get("optuna", {})
    if optuna.get("enabled", True):
        n_trials = optuna.get("n_trials", 50)
        if not isinstance(n_trials, int) or n_trials < 1:
            errors.append(f"model.optuna.n_trials = {n_trials!r} — 1 이상의 정수여야 합니다.")


def _validate_features(features: dict, errors: list) -> None:
    missing = features.get("missing", {})
    num_strat = missing.get("numeric_strategy", "median")
    if num_strat not in _VALID_NUMERIC_MISSING:
        errors.append(
            f"features.missing.numeric_strategy = '{num_strat}' — "
            f"허용 값: {sorted(_VALID_NUMERIC_MISSING)}"
        )

    cat_strat = missing.get("categorical_strategy", "mode")
    if cat_strat not in _VALID_CATEGORICAL_MISSING:
        errors.append(
            f"features.missing.categorical_strategy = '{cat_strat}' — "
            f"허용 값: {sorted(_VALID_CATEGORICAL_MISSING)}"
        )

    scaling = features.get("scaling", {})
    scale_method = scaling.get("method", "none")
    if scale_method not in _VALID_SCALING_METHODS:
        errors.append(
            f"features.scaling.method = '{scale_method}' — "
            f"허용 값: {sorted(_VALID_SCALING_METHODS)}"
        )

    encoding = features.get("encoding", {})
    enc_method = encoding.get("method", "none")
    if enc_method not in _VALID_ENCODING_METHODS:
        errors.append(
            f"features.encoding.method = '{enc_method}' — "
            f"허용 값: {sorted(_VALID_ENCODING_METHODS)}"
        )

    imbalance = features.get("imbalance", {})
    imb_method = imbalance.get("method", "none")
    if imb_method not in _VALID_IMBALANCE_METHODS:
        errors.append(
            f"features.imbalance.method = '{imb_method}' — "
            f"허용 값: {sorted(_VALID_IMBALANCE_METHODS)}"
        )


def _validate_evaluation(evaluation: dict, errors: list) -> None:
    metric = evaluation.get("optuna_target_metric", "logloss")
    if metric not in _VALID_OPTUNA_METRICS:
        errors.append(
            f"evaluation.optuna_target_metric = '{metric}' — "
            f"허용 값: {sorted(_VALID_OPTUNA_METRICS)}"
        )

    direction = evaluation.get("optuna_direction", "")
    if direction and direction not in _VALID_OPTUNA_DIRECTIONS:
        errors.append(
            f"evaluation.optuna_direction = '{direction}' — "
            f"허용 값: {sorted(_VALID_OPTUNA_DIRECTIONS)}"
        )


def _validate_paths(paths: dict, errors: list) -> None:
    required_path_keys = ["raw_data"]
    for key in required_path_keys:
        val = paths.get(key, "")
        if not val or not isinstance(val, str):
            errors.append(
                f"paths.{key} — 필수 경로 설정이 비어 있습니다. "
                "(CSV 또는 parquet 파일 경로를 입력해 주세요)"
            )
