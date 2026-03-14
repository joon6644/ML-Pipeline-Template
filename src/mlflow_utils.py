"""
mlflow_utils.py — MLflow 실험 트래킹 헬퍼
══════════════════════════════════════════
파이프라인 실행 결과(파라미터, 지표, 산출물)를 MLflow에 기록합니다.

사용 예시::

    from src.mlflow_utils import MLflowTracker

    tracker = MLflowTracker(config)
    with tracker.start_run(run_name="catboost_dev"):
        tracker.log_config_params()
        tracker.log_metrics({"f1_macro": 0.85, "roc_auc": 0.92})
        tracker.log_artifacts("results/figures")
        tracker.log_model_artifacts("models")

YAML 설정 예시::

    project:
      name: "my_project"
      mlflow_tracking_uri: "file:./mlruns"   # 로컬 저장 기본값
      mlflow_enabled: true                   # false면 모든 로깅 비활성화
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow 실험 추적을 관리하는 헬퍼 클래스.

    mlflow가 설치되지 않았거나 mlflow_enabled가 false인 경우
    모든 메서드가 조용히 무시(no-op)되어 기존 파이프라인에 영향을 주지 않습니다.

    Args:
        config: 전체 설정 dict (YAML에서 로드한 것)
    """

    def __init__(self, config: dict):
        self.config = config
        project_cfg = config.get("project", {})

        self._enabled: bool = project_cfg.get("mlflow_enabled", True)
        self._tracking_uri: str = project_cfg.get(
            "mlflow_tracking_uri", "file:./mlruns"
        )
        self._experiment_name: str = project_cfg.get("name", "ml_pipeline")
        self._mlflow = None
        self._active_run = None

        if self._enabled:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                mlflow.set_experiment(self._experiment_name)
                logger.info(
                    f"MLflow 초기화 완료 — experiment: '{self._experiment_name}', "
                    f"uri: '{self._tracking_uri}'"
                )
            except ImportError:
                logger.warning(
                    "mlflow 패키지가 설치되지 않아 실험 트래킹을 비활성화합니다. "
                    "(pip install mlflow)"
                )
                self._enabled = False

    @contextlib.contextmanager
    def start_run(self, run_name: str = "pipeline_run"):
        """MLflow Run을 컨텍스트 매니저로 시작합니다.

        사용 예시::

            with tracker.start_run(run_name="catboost_dev"):
                tracker.log_metrics({"f1_macro": 0.85})

        Args:
            run_name: MLflow UI에 표시될 실행 이름
        """
        if not self._enabled or self._mlflow is None:
            yield  # no-op
            return

        with self._mlflow.start_run(run_name=run_name) as run:
            self._active_run = run
            logger.info(f"MLflow Run 시작: '{run_name}' (run_id={run.info.run_id})")
            try:
                yield run
            finally:
                self._active_run = None
                logger.info(f"MLflow Run 종료: '{run_name}'")

    def log_config_params(self) -> None:
        """YAML 설정의 핵심 파라미터를 MLflow에 기록합니다.

        기록 항목:
        - project: task_type, seed
        - model: algorithm, n_splits, split_strategy, early_stopping_rounds
        - features: missing strategy, scaling method, encoding method, imbalance method
        - optuna: n_trials
        - evaluation: optuna_target_metric
        """
        if not self._enabled or self._mlflow is None:
            return

        project_cfg = self.config.get("project", {})
        model_cfg = self.config.get("model", {})
        feat_cfg = self.config.get("features", {})
        eval_cfg = self.config.get("evaluation", {})
        optuna_cfg = model_cfg.get("optuna", {})

        params = {
            # 프로젝트
            "task_type": project_cfg.get("task_type", "classification"),
            "seed": project_cfg.get("seed", 42),
            # 모델
            "algorithm": model_cfg.get("algorithm", "unknown"),
            "n_splits": model_cfg.get("n_splits", 5),
            "split_strategy": model_cfg.get("split_strategy", "stratified"),
            "early_stopping_rounds": model_cfg.get("early_stopping_rounds", 0),
            # 전처리
            "missing_numeric_strategy": feat_cfg.get("missing", {}).get(
                "numeric_strategy", "median"
            ),
            "missing_categorical_strategy": feat_cfg.get("missing", {}).get(
                "categorical_strategy", "mode"
            ),
            "scaling_method": feat_cfg.get("scaling", {}).get("method", "none"),
            "encoding_method": feat_cfg.get("encoding", {}).get("method", "none"),
            "imbalance_method": feat_cfg.get("imbalance", {}).get("method", "none"),
            # Optuna
            "optuna_enabled": optuna_cfg.get("enabled", True),
            "optuna_n_trials": optuna_cfg.get("n_trials", 50),
            "optuna_target_metric": eval_cfg.get("optuna_target_metric", "logloss"),
        }

        try:
            self._mlflow.log_params(params)
            logger.info(f"MLflow 파라미터 기록 완료 ({len(params)}개)")
        except Exception as e:
            logger.warning(f"MLflow 파라미터 기록 실패 (건너뜀): {e}")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """평가 지표를 MLflow에 기록합니다.

        Args:
            metrics: {"metric_name": value, ...} 형태의 지표 dict
        """
        if not self._enabled or self._mlflow is None or not metrics:
            return

        try:
            # MLflow는 None 값을 허용하지 않으므로 필터링
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean)
            logger.info(f"MLflow 지표 기록 완료 ({len(clean)}개): {clean}")
        except Exception as e:
            logger.warning(f"MLflow 지표 기록 실패 (건너뜀): {e}")

    def log_best_params(self, best_params: dict[str, Any] | None) -> None:
        """Optuna가 찾은 최적 하이퍼파라미터를 MLflow에 기록합니다.

        파라미터 이름에 'optuna_' 접두사를 붙여 일반 설정과 구분합니다.

        Args:
            best_params: Optuna 결과 파라미터 dict (None이면 건너뜀)
        """
        if not self._enabled or self._mlflow is None or not best_params:
            return

        try:
            prefixed = {f"optuna_{k}": v for k, v in best_params.items()}
            self._mlflow.log_params(prefixed)
            logger.info(f"MLflow Optuna 최적 파라미터 기록 완료 ({len(prefixed)}개)")
        except Exception as e:
            logger.warning(f"MLflow Optuna 파라미터 기록 실패 (건너뜀): {e}")

    def log_threshold(self, threshold: float | None) -> None:
        """최적 임계값을 MLflow에 기록합니다.

        Args:
            threshold: 최적 임계값 (None이면 건너뜀)
        """
        if not self._enabled or self._mlflow is None or threshold is None:
            return

        try:
            self._mlflow.log_param("best_threshold", threshold)
            logger.info(f"MLflow 임계값 기록 완료: {threshold:.4f}")
        except Exception as e:
            logger.warning(f"MLflow 임계값 기록 실패 (건너뜀): {e}")

    def log_artifacts(self, artifact_dir: str | Path) -> None:
        """지정된 폴더의 모든 파일을 MLflow 아티팩트로 기록합니다.

        Args:
            artifact_dir: 기록할 폴더 경로 (results/figures 등)
        """
        if not self._enabled or self._mlflow is None:
            return

        artifact_dir = Path(artifact_dir)
        if not artifact_dir.exists():
            logger.warning(f"MLflow 아티팩트 폴더 없음 (건너뜀): {artifact_dir}")
            return

        try:
            self._mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_dir.name)
            logger.info(f"MLflow 아티팩트 기록 완료: {artifact_dir}")
        except Exception as e:
            logger.warning(f"MLflow 아티팩트 기록 실패 (건너뜀): {e}")

    def log_model_artifacts(self, model_dir: str | Path) -> None:
        """모델 폴더의 파일들을 MLflow 아티팩트로 기록합니다.

        Args:
            model_dir: 모델 저장 폴더 경로 (models/ 등)
        """
        if not self._enabled or self._mlflow is None:
            return

        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.warning(f"MLflow 모델 폴더 없음 (건너뜀): {model_dir}")
            return

        try:
            self._mlflow.log_artifacts(str(model_dir), artifact_path="models")
            logger.info(f"MLflow 모델 아티팩트 기록 완료: {model_dir}")
        except Exception as e:
            logger.warning(f"MLflow 모델 아티팩트 기록 실패 (건너뜀): {e}")

    @property
    def is_enabled(self) -> bool:
        """MLflow 추적이 활성화되어 있는지 반환합니다."""
        return self._enabled
