"""
ensemble.py — 저장된 여러 모델들의 예측값을 결합(앙상블)하는 독립 스크립트
════════════════════════════════════════════════════════════════════
파이프라인 외부에서 단독으로 실행 가능하며,
CatBoost, XGBoost, LightGBM, RF 등 `--final-eval`로 저장 완료된
단일 모델들의 결과를 모아 가중 평균(Soft/Hard Voting)을 수행합니다.

사용법:
    python -m src.ensemble --data data/raw/test.csv \\
                           --models catboost_final.cbm xgboost_final.pkl \\
                           --weights 0.6 0.4 \\
                           --output results/submissions/ensemble_sub.csv
"""

import argparse
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# CatBoost는 전용 load_model이 필요함
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    pass


logger = logging.getLogger("ensemble")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(description="다중 모델 앙상블(블렌딩) 예측 스크립트")
    parser.add_argument("--data", type=str, required=True, help="추론할 대상 CSV 파일 경로")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="단일 모델 경로 리스트 (models/ 폴더 기준 또는 절대경로) 예: catboost_final.cbm rf_final.pkl"
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="각 모델별 앙상블 가중치 리스트 (입력 생략 시 1/N 동일 가중치 적용) 예: 0.5 0.3 0.2"
    )
    parser.add_argument("--dp", type=str, default="models/data_preprocessor.pkl", help="사용할 DataPreprocessor (기본값: models/data_preprocessor.pkl)")
    parser.add_argument("--encoders", type=str, default="models/label_encoders.pkl", help="사용할 LabelEncoders (있을 경우만 적용)")
    parser.add_argument("--output", type=str, default="results/submissions/ensemble_submission.csv", help="저장할 CSV 파일 경로")
    
    return parser.parse_args()


def load_model(path: Path):
    """모델 확장자에 따라 적절한 로더를 선택합니다."""
    path_str = str(path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path_str}")
        
    if path_str.endswith(".cbm"):
        logger.info(f"Loading CatBoost model from {path.name}...")
        try:
            model = CatBoostClassifier()
            model.load_model(path_str)
            return model, "classification"
        except Exception:
            model = CatBoostRegressor()
            model.load_model(path_str)
            return model, "regression"
    else:
        logger.info(f"Loading joblib/pickle model from {path.name}...")
        model = joblib.load(path_str)
        # 단순 분류기 여부 판별
        task_type = "classification" if hasattr(model, "predict_proba") else "regression"
        return model, task_type


def main():
    args = parse_args()

    # 1. 제약 조건 검사
    if args.weights and len(args.models) != len(args.weights):
        logger.error("모델 개수와 옵션 가중치(--weights) 개수가 일치하지 않습니다!")
        sys.exit(1)

    weights = args.weights
    if not weights:
        w = 1.0 / len(args.models)
        weights = [w] * len(args.models)
    else:
        # 가중치 합이 1이 되도록 정규화
        total = sum(weights)
        weights = [w / total for w in weights]
        
    logger.info(f"앙상블 시작: 모델 {len(args.models)}개")
    logger.info(f"적용 가중치: {weights}")

    # 2. 데이터 및 전처리기 로드
    df = pd.read_csv(args.data)
    logger.info(f"원본 데이터 크기: {df.shape}")

    dp_path = Path(args.dp)
    if not dp_path.exists():
        logger.error(f"DataPreprocessor를 찾을 수 없습니다: {dp_path}")
        sys.exit(1)
        
    dp = joblib.load(dp_path)
    logger.info("DataPreprocessor 변환 적용 중...")
    X_transformed = dp.transform(df)

    # LabelEncoders 로드 및 적용 (선택)
    enc_path = Path(args.encoders)
    if enc_path.exists():
        encoders = joblib.load(enc_path)
        logger.info(f"LabelEncoders 적용 중 ({len(encoders)}개 변수)...")
        X_transformed = X_transformed.copy()
        for col, le in encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype(str).map(
                    lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
                )

    # 3. 모델별 예측 획득
    all_preds_proba = []
    
    for _idx, model_path in enumerate(args.models):
        path = Path(model_path) if Path(model_path).is_absolute() else Path("models") / model_path
        model, task_type = load_model(path)

        # 예측 실행
        if task_type == "classification":
            proba = model.predict_proba(X_transformed)
            # 이진 분류 vs 다중 분류 처리
            if proba.shape[1] == 2:
                preds = proba[:, 1]
            else:
                preds = proba
        else:
            preds = model.predict(X_transformed)
            
        all_preds_proba.append(preds * weights[_idx])

    # 4. 가중 평균 합산 블렌딩 (Soft Voting)
    final_preds_proba = np.sum(all_preds_proba, axis=0)

    # 5. 최종 결정 및 제출 파일 생성 (이진 예측의 경우 임계치 0.5 하드 코딩, 추후 보강 가능)
    if final_preds_proba.ndim == 1:
        # 회귀 또는 이진분류
        is_classification = any(hasattr(model, "predict_proba") for model, _ in [load_model(Path(p) if Path(p).is_absolute() else Path("models") / p) for p in args.models[:1]])
        if is_classification:
            final_class = (final_preds_proba >= 0.5).astype(int)
        else:
            final_class = final_preds_proba # 회귀
    else:
        # 다중 분류 (확률이 가장 높은 클래스)
        final_class = np.argmax(final_preds_proba, axis=1)

    out_df = pd.DataFrame({
        "pred_proba": final_preds_proba if final_preds_proba.ndim == 1 else final_preds_proba.max(axis=1),
        "prediction": final_class
    })

    # 원본 id 유지 시도 (id 컬럼이 있을 경우)
    for maybe_id in ["id", "ID", "index", "custId"]:
        if maybe_id in df.columns:
            out_df.insert(0, maybe_id, df[maybe_id].values)
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    logger.info("=" * 60)
    logger.info(f"🎉 앙상블 완료! 제출 파일 생성됨: {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
