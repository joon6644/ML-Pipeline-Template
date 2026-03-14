"""
predict.py — 저장된 모델로 새 데이터를 예측하는 추론 스크립트
═══════════════════════════════════════════════════════════════
학습이 완료된 후 models/ 폴더에 저장된 모델과 전처리기를 로드하여,
새로운 데이터 파일(CSV)에 대한 예측 결과를 생성합니다.

사용법::

    # 기본 사용 (results/submissions/ 에 예측 결과 저장)
    python predict.py --config configs/catboost.yaml --input data/raw/test.csv

    # 결과 파일 경로 직접 지정
    python predict.py --config configs/catboost.yaml --input data/raw/test.csv --output results/my_prediction.csv

    # 이진 분류에서 임계값 지정
    python predict.py --config configs/catboost.yaml --input data/raw/test.csv --threshold 0.45

준비 조건::

    - run_pipeline.py --final-eval 을 먼저 실행하여 다음 파일이 존재해야 합니다:
      · models/{algorithm}_final.pkl (또는 .cbm)
      · models/data_preprocessor.pkl
      · models/label_encoders.pkl    (XGBoost/RF 등 사용 시)
      · models/best_threshold.pkl    (--tune-threshold 사용 시)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, get_paths, get_model_params, get_feature_config
from src.utils import get_logger, load_dataframe, ensure_dir

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🔮 저장된 ML 모델로 새 데이터를 예측합니다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str, required=True,
        help="학습에 사용했던 YAML 설정 파일 경로 (예: configs/catboost.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str, required=True,
        help="예측할 새 데이터 CSV 파일 경로",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="예측 결과 저장 경로 (기본값: results/submissions/prediction.csv)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float, default=None,
        help=(
            "이진 분류 결정 임계값 (기본값: models/best_threshold.pkl 또는 0.5)\n"
            "예: --threshold 0.45"
        ),
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="예측 확률도 함께 출력합니다 (이진: positive 확률, 다중: 클래스별 확률)",
    )
    return parser.parse_args()


def load_artifacts(config: dict) -> tuple:
    """모델, 전처리기, 인코더, 임계값을 로드합니다.

    Returns:
        (model, dp, encoders, best_threshold)
    """
    paths = get_paths(config)
    model_cfg = get_model_params(config)
    model_dir = Path(paths.get("model_dir", "models"))
    algorithm = model_cfg["algorithm"]

    # ── 모델 로드 ──
    if algorithm == "catboost":
        cbm_path = model_dir / f"{algorithm}_final.cbm"
        if not cbm_path.exists():
            raise FileNotFoundError(
                f"최종 모델 파일이 없습니다: {cbm_path}\n"
                "먼저 'python run_pipeline.py --config ... --final-eval'을 실행해 주세요."
            )
        from catboost import CatBoostClassifier, CatBoostRegressor
        task_type = model_cfg["task_type"]
        ModelClass = CatBoostRegressor if task_type == "regression" else CatBoostClassifier
        model = ModelClass()
        model.load_model(str(cbm_path))
        logger.info(f"CatBoost 모델 로드: {cbm_path}")
    else:
        pkl_path = model_dir / f"{algorithm}_final.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"최종 모델 파일이 없습니다: {pkl_path}\n"
                "먼저 'python run_pipeline.py --config ... --final-eval'을 실행해 주세요."
            )
        model = joblib.load(pkl_path)
        logger.info(f"모델 로드: {pkl_path}")

    # ── DataPreprocessor 로드 ──
    dp_path = model_dir / "data_preprocessor.pkl"
    if not dp_path.exists():
        raise FileNotFoundError(
            f"전처리기 파일이 없습니다: {dp_path}\n"
            "먼저 'python run_pipeline.py --config ... --final-eval'을 실행해 주세요."
        )
    dp = joblib.load(dp_path)
    logger.info(f"DataPreprocessor 로드: {dp_path}")

    # ── LabelEncoders 로드 (있으면) ──
    enc_path = model_dir / "label_encoders.pkl"
    encoders = joblib.load(enc_path) if enc_path.exists() else {}
    if encoders:
        logger.info(f"LabelEncoders 로드: {enc_path} ({len(encoders)}개 컬럼)")

    # ── 최적 임계값 로드 (있으면) ──
    thr_path = model_dir / "best_threshold.pkl"
    best_threshold = joblib.load(thr_path) if thr_path.exists() else None
    if best_threshold is not None:
        logger.info(f"저장된 최적 임계값 로드: {best_threshold:.4f}")

    return model, dp, encoders, best_threshold


def preprocess_input(
    df: pd.DataFrame,
    dp,
    encoders: dict,
    config: dict,
) -> pd.DataFrame:
    """입력 데이터에 훈련 때와 동일한 전처리를 적용합니다."""
    data_cfg = config.get("data", {})
    feat_cfg = get_feature_config(config)
    target_col = data_cfg.get("target_col", "target")
    id_col = data_cfg.get("id_col")
    drop_cols = feat_cfg.get("drop_cols", [])

    # ID 컬럼 보존 (결과 파일에 붙여주기 위해)
    id_series = None
    if id_col and id_col in df.columns:
        id_series = df[id_col].copy()

    # 타겟 컬럼이 있으면 제거 (test.csv에 있는 경우)
    cols_to_drop = [c for c in [target_col] + drop_cols if c in df.columns]
    if id_col and id_col in df.columns:
        cols_to_drop.append(id_col)
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 훈련 때 학습된 전처리 통계량으로 transform (fit 없이!)
    df = dp.transform(df)

    # LabelEncoder 적용 (XGBoost/RF 등)
    if encoders:
        df = df.copy()
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x, _le=le: (
                        _le.transform([x])[0] if x in _le.classes_ else -1
                    )
                )

    return df, id_series


def predict(
    model,
    X: pd.DataFrame,
    config: dict,
    threshold: float | None,
    include_proba: bool,
) -> pd.DataFrame:
    """모델로 예측하고 결과 DataFrame을 반환합니다."""
    model_cfg = get_model_params(config)
    task_type = model_cfg["task_type"]

    result = {}

    if task_type == "regression":
        result["prediction"] = model.predict(X)
    else:
        # 분류: 확률 예측 시도
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            n_classes = proba.shape[1] if proba.ndim == 2 else 2
            is_binary = (n_classes == 2) or (proba.ndim == 1)

            if is_binary:
                proba_pos = proba[:, 1] if proba.ndim == 2 else proba
                thr = threshold if threshold is not None else 0.5
                result["prediction"] = (proba_pos >= thr).astype(int)
                if include_proba:
                    result["proba_positive"] = proba_pos
                logger.info(f"이진 분류 예측 완료 (임계값: {thr:.4f})")
            else:
                # 다중 분류
                result["prediction"] = np.argmax(proba, axis=1)
                if include_proba:
                    for i in range(n_classes):
                        result[f"proba_class_{i}"] = proba[:, i]
                logger.info(f"다중 분류({n_classes}클래스) 예측 완료")
        else:
            result["prediction"] = model.predict(X)
            logger.info("predict_proba 없는 모델 — predict()로 예측 완료")

    return pd.DataFrame(result)


def main() -> None:
    args = parse_args()

    # ── 설정 로드 ──
    config = load_config(args.config)
    paths = get_paths(config)
    data_cfg = config.get("data", {})

    logger.info("=" * 60)
    logger.info("  🔮 추론(Inference) 시작")
    logger.info(f"  설정 파일: {args.config}")
    logger.info(f"  입력 데이터: {args.input}")
    logger.info("=" * 60)

    # ── 입력 데이터 로드 ──
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 파일이 없습니다: {input_path}")
        sys.exit(1)

    sep = data_cfg.get("separator", ",")
    enc = data_cfg.get("encoding", "utf-8")
    if input_path.suffix in (".parquet",):
        df_input = pd.read_parquet(input_path)
    else:
        df_input = pd.read_csv(input_path, sep=sep, encoding=enc)

    logger.info(f"입력 데이터 로드 완료: {df_input.shape}")

    # ── 모델/전처리기 로드 ──
    model, dp, encoders, saved_threshold = load_artifacts(config)

    # 임계값 우선순위: CLI 인수 > 저장된 값 > 기본값(0.5)
    threshold = args.threshold if args.threshold is not None else saved_threshold

    # ── 전처리 ──
    X, id_series = preprocess_input(df_input.copy(), dp, encoders, config)
    logger.info(f"전처리 완료: {X.shape}")

    # ── 예측 ──
    pred_df = predict(model, X, config, threshold, args.proba)

    # ── ID 컬럼 붙이기 ──
    id_col = data_cfg.get("id_col")
    if id_series is not None and id_col:
        pred_df.insert(0, id_col, id_series.values)

    # ── 결과 저장 ──
    if args.output:
        output_path = Path(args.output)
    else:
        submission_dir = paths.get("submission_dir", "results/submissions")
        ensure_dir(submission_dir)
        algorithm = config.get("model", {}).get("algorithm", "model")
        output_path = Path(submission_dir) / f"{algorithm}_prediction.csv"

    pred_df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("=" * 60)
    logger.info(f"  ✅ 예측 완료! 결과 저장: {output_path}")
    logger.info(f"  예측 건수: {len(pred_df):,}행")
    logger.info("=" * 60)

    # 미리보기 출력
    print("\n📋 예측 결과 미리보기 (상위 5행):")
    print(pred_df.head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
