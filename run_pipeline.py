"""
run_pipeline.py — ML 파이프라인 메인 실행 스크립트
═══════════════════════════════════════════════════
3가지 모드로 실행할 수 있습니다:

[1단계] 초기 분할:
    python run_pipeline.py --config configs/catboost.yaml --split-holdout 0.2

[2~5단계] K-Fold 개발 (기본 모드):
    python run_pipeline.py --config configs/catboost.yaml
    python run_pipeline.py --config configs/catboost.yaml --skip-tuning
    python run_pipeline.py --config configs/catboost.yaml --skip-preprocess

[6단계] 최종 평가:
    python run_pipeline.py --config configs/catboost.yaml --final-eval

⚠️ 데이터 누수 방지: 결측치 처리(median/mean/mode)는 전처리 단계가 아니라
   trainer의 K-Fold 루프 내부에서 DataPreprocessor(fit/transform)로 수행됩니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, get_paths, get_model_params, get_feature_config
from src.utils import set_seed, get_logger, load_dataframe, save_dataframe, Timer
from src.preprocessor import preprocess_pipeline, split_holdout, DataPreprocessor
from src.trainer import run_optuna_tuning, train_final_model, train_full_model
from src.evaluator import evaluate_and_visualize, calculate_metrics, generate_report, find_best_threshold
from src.mlflow_utils import MLflowTracker


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="🚀 트리 모델 기반 ML 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="YAML 설정 파일 경로 (예: configs/catboost.yaml)",
    )
    parser.add_argument(
        "--split-holdout",
        type=float,
        default=None,
        metavar="RATIO",
        help="[1단계] 데이터를 훈련/홀드아웃으로 분할합니다.\n"
             "비율을 인자로 전달 (예: 0.2 = 20%% 홀드아웃).\n"
             "이 모드에서는 전처리 → 분할 → 저장만 수행합니다.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="전처리를 건너뛰고 이미 저장된 interim/train 데이터를 사용합니다.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Optuna 튜닝을 건너뛰고 고정 파라미터만으로 학습합니다.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="[6단계] 전체 훈련 데이터로 최종 모델을 학습하고,\n"
             "홀드아웃 데이터로 최종 평가를 수행합니다.",
    )
    parser.add_argument(
        "--tune-threshold",
        type=str,
        default=None,
        metavar="METRIC",
        help="이진 분류에서 주어진 지표(예: f1_macro)를 기준 최적의 Threshold를 찾습니다.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="실험 실행 ID. 입력하지 않으면 현재 시각으로 자동 생성 (결과 덮어쓰기 방지).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
#  모드 1: 초기 Hold-out 분할
# ═══════════════════════════════════════════════════════════════
def run_split_holdout(config: dict, holdout_ratio: float, logger) -> None:
    """전처리 → Hold-out 분할 → interim/ 저장."""
    logger.info("=" * 60)
    logger.info(f"  📦 [1단계] Hold-out 분할 (ratio={holdout_ratio})")
    logger.info("=" * 60)

    with Timer("전처리 + Hold-out 분할", logger):
        df = preprocess_pipeline(config)
        train_df, holdout_df = split_holdout(df, config, holdout_ratio)

    logger.info("=" * 60)
    logger.info("  ✅ Hold-out 분할 완료!")
    logger.info("    이후 개발은 interim/train 데이터만 사용합니다.")
    logger.info("    홀드아웃은 --final-eval 시에만 사용됩니다.")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  모드 2: K-Fold 개발 (기본 모드)
# ═══════════════════════════════════════════════════════════════
def run_development(config: dict, args, logger) -> None:
    """전처리(또는 로드) → Optuna 튜닝 → K-Fold 학습 → OOF 평가."""
    paths = get_paths(config)
    model_cfg = get_model_params(config)
    data_cfg = config.get("data", {})

    target_col = data_cfg.get("target_col", "target")
    id_col = data_cfg.get("id_col")
    algorithm = model_cfg["algorithm"]      # MLflow에서 사용
    task_type = model_cfg["task_type"]      # MLflow에서 사용

    # ── run-id: 타임스탬프 기반 버전 관리 ──
    if args.run_id:
        run_id = args.run_id
    else:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M")

    # run_id를 결과/모델 경로에 반영 (원본 paths 변경 없이 로컬 변수로 override)
    figure_dir_orig = paths.get("figure_dir", "results/figures")
    result_dir_orig = paths.get("result_dir", "results")
    model_dir_orig = paths.get("model_dir", "models")

    paths_local = dict(paths)
    paths_local["figure_dir"] = f"{figure_dir_orig}/{run_id}"
    paths_local["result_dir"] = result_dir_orig  # report는 공용
    paths_local["model_dir"] = model_dir_orig     # 모델은 공용 (predict.py 호환)

    from src.utils import ensure_dir
    ensure_dir(paths_local["figure_dir"])

    logger.info(f"실행 ID: {run_id} (그래프: {paths_local['figure_dir']}/)")

    logger.info("=" * 60)
    logger.info("  🔬 [2~5단계] K-Fold 개발 모드")
    logger.info("=" * 60)

    # ── 데이터 로드 ──
    if args.skip_preprocess:
        # interim/train이 있으면 사용, 없으면 processed 사용
        interim_train = paths.get("interim_train", "data/interim/train.parquet")
        processed_path = paths.get("processed_data", "data/processed/train_processed.parquet")

        if Path(interim_train).exists():
            logger.info(f"interim 훈련 데이터 로드: {interim_train}")
            df = load_dataframe(interim_train)
        elif Path(processed_path).exists():
            logger.info(f"processed 데이터 로드: {processed_path}")
            df = load_dataframe(processed_path)
        else:
            raise FileNotFoundError(
                f"로드할 데이터가 없습니다. "
                f"먼저 --split-holdout 또는 전처리를 실행하세요."
            )
    else:
        # 전처리 실행
        with Timer("전체 전처리", logger):
            df = preprocess_pipeline(config)

        # interim/train이 있으면 훈련용만 사용 (홀드아웃 제외)
        interim_train = paths.get("interim_train", "data/interim/train.parquet")
        if Path(interim_train).exists():
            logger.info(f"⚠️ interim/train 존재 → 훈련 데이터만 사용 (홀드아웃 보호)")
            df = load_dataframe(interim_train)

    # ── X / y 분리 ──
    exclude_cols = [target_col]
    if id_col and id_col in df.columns:
        exclude_cols.append(id_col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"피처 수: {X.shape[1]}, 샘플 수: {X.shape[0]}")
    logger.info(f"타겟 분포:\n{y.value_counts().to_string()}")

    # ── MLflow 실험 트래킹 ──
    tracker = MLflowTracker(config)

    with tracker.start_run(run_name=f"{algorithm}_dev"):
        # 설정 파라미터 기록
        tracker.log_config_params()

        # ── Optuna 튜닝 ──
        best_params = None
        optuna_cfg = model_cfg.get("optuna", {})

        if not args.skip_tuning and optuna_cfg.get("enabled", True):
            with Timer("Optuna 튜닝", logger):
                best_params = run_optuna_tuning(X, y, config)
        else:
            logger.info("Optuna 튜닝 건너뜀 — 고정 파라미터로 학습합니다.")

        # ── K-Fold 학습 ──
        with Timer("K-Fold 학습", logger):
            result = train_final_model(X, y, config, best_params)

        # ── --tune-threshold: 임계값 수동 튜닝 ──
        best_threshold = None
        if args.tune_threshold and task_type == "classification":
            logger.info(f"임계값 튜닝 시작 (목표 지표: {args.tune_threshold})...")
            best_threshold, best_score = find_best_threshold(
                y_true=y.values,
                y_proba_positive=result.oof_preds,
                metric=args.tune_threshold,
                search_range=(0.1, 0.9),
                search_step=0.01,
            )
            # OOF 예측 업데이트 (최적 임계값 적용)
            result.oof_preds = (result.oof_preds >= best_threshold).astype(int)

            # 임계값 저장 (최종 평가용)
            threshold_path = Path(paths.get("model_dir", "models")) / "best_threshold.pkl"
            joblib.dump(best_threshold, threshold_path)
            logger.info(f"Best threshold 저장: {threshold_path}")

        # ── OOF 평가 & 시각화 ──
        with Timer("평가 & 시각화", logger):
            metrics = evaluate_and_visualize(
                y_true=y.values,
                oof_preds=result.oof_preds,
                models=result.models,
                feature_names=result.feature_names,
                X=X,
                config=config,
                best_params=best_params,
                best_threshold=best_threshold,
            )

        # ── Fold별 성능 박스플롯 시각화 ──
        if result.fold_scores:
            from src.evaluator import plot_fold_scores
            eval_cfg = config.get("evaluation", {})
            metric_name = eval_cfg.get("optuna_target_metric", "score")
            figure_dir = Path(paths.get("figure_dir", "results/figures"))
            plot_fold_scores(
                result.fold_scores,
                figure_dir / "fold_scores.png",
                metric_name=metric_name,
                dpi=config.get("visualization", {}).get("dpi", 150),
            )

        # ── best_params 저장 (최종 평가 시 재사용) ──
        if best_params:
            params_path = Path(paths.get("model_dir", "models")) / "best_params.pkl"
            joblib.dump(best_params, params_path)
            logger.info(f"Best params 저장: {params_path}")

        # ── MLflow: 지표 / Optuna 파라미터 / 아티팩트 기록 ──
        tracker.log_metrics(metrics)
        tracker.log_best_params(best_params)
        tracker.log_threshold(best_threshold)
        tracker.log_artifacts(paths.get("figure_dir", "results/figures"))
        tracker.log_artifacts(paths.get("result_dir", "results"))
        tracker.log_model_artifacts(paths.get("model_dir", "models"))

    logger.info("=" * 60)
    logger.info("  ✅ K-Fold 개발 완료!")
    logger.info("  만족스러우면 --final-eval로 최종 평가를 진행하세요.")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  모드 3: 최종 평가 (Hold-out)
# ═══════════════════════════════════════════════════════════════
def run_final_eval(config: dict, args, logger) -> None:
    """전체 훈련 데이터로 최종 모델 학습 → 홀드아웃 평가 → 결과 저장."""
    paths = get_paths(config)
    model_cfg = get_model_params(config)
    feat_cfg = get_feature_config(config)
    data_cfg = config.get("data", {})

    target_col = data_cfg.get("target_col", "target")
    id_col = data_cfg.get("id_col")
    algorithm = model_cfg["algorithm"]
    task_type = model_cfg["task_type"]

    # ── run-id: 타임스탬프 기반 버전 관리 ──
    if args.run_id:
        run_id = args.run_id
    else:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M")

    figure_dir_orig = paths.get("figure_dir", "results/figures")
    result_dir_orig = paths.get("result_dir", "results")
    model_dir_orig = paths.get("model_dir", "models")

    paths_local = dict(paths)
    paths_local["figure_dir"] = f"{figure_dir_orig}/{run_id}"
    paths_local["result_dir"] = result_dir_orig
    paths_local["model_dir"] = model_dir_orig

    from src.utils import ensure_dir
    ensure_dir(paths_local["figure_dir"])

    logger.info("=" * 60)
    logger.info(f"  🏆 [6단계] 최종 평가 모드 (Run ID: {run_id})")
    logger.info("=" * 60)

    # ── 훈련/홀드아웃 데이터 로드 ──
    interim_train = paths.get("interim_train", "data/interim/train.parquet")
    interim_holdout = paths.get("interim_holdout", "data/interim/holdout.parquet")

    if not Path(interim_train).exists() or not Path(interim_holdout).exists():
        raise FileNotFoundError(
            f"interim 데이터가 없습니다. "
            f"먼저 --split-holdout으로 데이터를 분할하세요.\n"
            f"  필요 파일: {interim_train}, {interim_holdout}"
        )

    train_df = load_dataframe(interim_train)
    holdout_df = load_dataframe(interim_holdout)
    logger.info(f"훈련 데이터: {train_df.shape}, 홀드아웃 데이터: {holdout_df.shape}")

    # ── X / y 분리 ──
    exclude_cols = [target_col]
    if id_col and id_col in train_df.columns:
        exclude_cols.append(id_col)

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[target_col]

    # ── MLflow 실험 트래킹 ──
    tracker = MLflowTracker(config)

    # ── 저장된 best_params 로드 (있으면) ──
    params_path = Path(paths_local["model_dir"]) / "best_params.pkl"
    best_params = None
    if params_path.exists():
        best_params = joblib.load(params_path)
        logger.info(f"Best params 로드: {best_params}")
    else:
        logger.info("저장된 best_params 없음 — 고정 파라미터로 학습합니다.")

    # ── 전체 훈련 데이터로 최종 모델 학습 ──
    model, dp, encoders = train_full_model(X_train, y_train, config, best_params)

    # ── 홀드아웃 데이터 전처리 (훈련 통계량 적용) ──
    logger.info("홀드아웃 데이터 전처리 (훈련 통계량 적용)...")
    X_holdout_transformed = dp.transform(X_holdout)

    # 범주형 인코딩 (XGBoost · RandomForest)
    if encoders:
        X_holdout_transformed = X_holdout_transformed.copy()
        for col, le in encoders.items():
            if col in X_holdout_transformed.columns:
                X_holdout_transformed[col] = X_holdout_transformed[col].astype(str).map(
                    lambda x, _le=le: (
                        _le.transform([x])[0] if x in _le.classes_ else -1
                    )
                )

    # ── 홀드아웃 예측 ──
    best_threshold = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        holdout_proba = model.predict_proba(X_holdout_transformed)
        n_classes = y_holdout.nunique()
        if n_classes <= 2:
            holdout_preds_raw = holdout_proba[:, 1] if holdout_proba.ndim == 2 else holdout_proba
            
            # ── 임계값 수동 튜닝이 저장되어 있으면 적용 ──
            threshold_path = Path(paths_local["model_dir"]) / "best_threshold.pkl"
            if threshold_path.exists():
                best_threshold = joblib.load(threshold_path)
                logger.info(f"저장된 Best threshold ({best_threshold:.4f}) 로드 및 적용")
                holdout_preds_raw = (holdout_preds_raw >= best_threshold).astype(int)
        else:
            holdout_preds_raw = holdout_proba
    else:
        holdout_preds_raw = model.predict(X_holdout_transformed)

    # ── 홀드아웃 평가 ──
    with tracker.start_run(run_name=f"{algorithm}_final"):
        # 설정 파라미터 기록
        tracker.log_config_params()
        tracker.log_best_params(best_params)

        with Timer("홀드아웃 평가 & 시각화", logger):
            metrics = evaluate_and_visualize(
                y_true=y_holdout.values,
                oof_preds=holdout_preds_raw,
                models=[model],
                feature_names=feature_cols,
                X=X_holdout_transformed,
                config=config,
                best_params=best_params,
                best_threshold=best_threshold,
            )

        # ── MLflow: 지표 / 임계값 / 아티팩트 기록 ──
        tracker.log_metrics(metrics)
        tracker.log_threshold(best_threshold)
        tracker.log_artifacts(paths_local["figure_dir"])
        tracker.log_artifacts(paths_local["result_dir"])
        tracker.log_model_artifacts(paths_local["model_dir"])

    logger.info("=" * 60)
    logger.info("  🏆 최종 평가 완료!")
    logger.info(f"  최종 모델: {paths_local['model_dir']}/{algorithm}_final.*")
    logger.info(f"  평가 결과: {paths_local['result_dir']}/evaluation_report.txt")
    logger.info(f"  시각화 차트: {paths_local['figure_dir']}/")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    """파이프라인 메인 함수."""
    args = parse_args()

    # 설정 로드
    config = load_config(args.config)
    project_cfg = config.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    logger = get_logger("pipeline", log_file="results/pipeline.log")
    logger.info(f"🚀 파이프라인 시작: {project_cfg.get('name', 'unknown')}")
    logger.info(f"설정 파일: {args.config}")

    # 모드 분기
    if args.split_holdout is not None:
        run_split_holdout(config, args.split_holdout, logger)
    elif args.final_eval:
        run_final_eval(config, args, logger)
    else:
        run_development(config, args, logger)


if __name__ == "__main__":
    main()
