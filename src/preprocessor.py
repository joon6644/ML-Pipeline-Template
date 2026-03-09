"""
preprocessor.py — 데이터 정제 및 피처 엔지니어링
═══════════════════════════════════════════════════
1단계 전처리: 원본 CSV 로드 → 컬럼 제거 → 피처 생성 → parquet 저장
2단계 전처리: DataPreprocessor (fit/transform) → K-Fold 내부에서 결측치 처리

⚠️ 결측치 처리(median/mean/mode)는 DataPreprocessor.fit()을 통해
   훈련 데이터만의 통계량으로 수행되어 데이터 누수를 방지합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_feature_config, get_paths
from src.utils import ensure_dir, get_logger, load_dataframe, save_dataframe, Timer

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  DataPreprocessor — fit/transform 패턴 (데이터 누수 방지)
# ═══════════════════════════════════════════════════════════════
class DataPreprocessor:
    """훈련 데이터 기준의 통계량으로 결측치 처리 + sklearn 변환을 수행하는 전처리기.
    
    데이터 누수를 방지하기 위해 반드시 **훈련 데이터로만 fit()** 하고,
    훈련·검증·테스트 데이터 모두에 **동일한 통계량으로 transform()** 합니다.
    
    sklearn의 Scaler/Encoder 객체를 내부에서 사용하므로,
    fit 시 학습된 파라미터(mean_, scale_, categories_ 등)가 자동 저장됩니다.
    
    사용 예시::
    
        preprocessor = DataPreprocessor(config)
        X_train = preprocessor.fit_transform(X_train)  # 훈련 데이터로 fit + transform
        X_val   = preprocessor.transform(X_val)         # 훈련 통계량으로 transform
        X_test  = preprocessor.transform(X_test)        # 훈련 통계량으로 transform
    
    YAML 설정 예시::
    
        features:
          scaling:
            method: "standard"   # standard | robust | minmax | none
            cols: "auto"         # auto = num_cols, 또는 명시적 리스트
          encoding:
            method: "ordinal"    # ordinal | label | none
            cols: "auto"         # auto = cat_cols, 또는 명시적 리스트
    """

    # ── 지원하는 sklearn 변환기 매핑 ──────────────────────────────
    _SCALER_MAP = {
        "standard": "sklearn.preprocessing.StandardScaler",
        "robust": "sklearn.preprocessing.RobustScaler",
        "minmax": "sklearn.preprocessing.MinMaxScaler",
    }
    _ENCODER_MAP = {
        "ordinal": "sklearn.preprocessing.OrdinalEncoder",
        "label": "sklearn.preprocessing.LabelEncoder",
        "target": "category_encoders.target_encoder.TargetEncoder",
    }

    def __init__(self, config: dict):
        """
        Args:
            config: 전체 설정 dict (YAML에서 로드한 것)
        """
        self.config = config
        self._feat_cfg = get_feature_config(config)
        self._data_cfg = config.get("data", {})
        self._missing_cfg = self._feat_cfg.get("missing", {})
        self._scaling_cfg = self._feat_cfg.get("scaling", {})
        self._encoding_cfg = self._feat_cfg.get("encoding", {})

        # fit()에서 저장될 통계량 및 Imputer 객체
        self._num_fill_values: dict[str, float] = {}   # {col_name: median/mean/zero 값}
        self._cat_fill_values: dict[str, str] = {}     # {col_name: mode/constant 값}
        self._num_imputer = None                       # KNNImputer / IterativeImputer

        # sklearn 변환기 (fit 후 파라미터가 자동 저장됨)
        self._scaler = None          # StandardScaler / RobustScaler / MinMaxScaler
        self._encoder = None         # OrdinalEncoder
        self._scaler_cols: list[str] = []    # 스케일링 대상 컬럼
        self._encoder_cols: list[str] = []   # 인코딩 대상 컬럼

        self._is_fitted: bool = False

    @staticmethod
    def _import_class(dotted_path: str):
        """문자열 경로에서 클래스를 동적 임포트합니다."""
        module_path, class_name = dotted_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _validate_cols(cols: list[str], df: pd.DataFrame, context: str) -> list[str]:
        """사용자가 지정한 컬럼명이 실제 데이터에 존재하는지 검증합니다.

        존재하지 않는 컬럼이 하나라도 있으면 ValueError를 발생시켜,
        오타나 삭제된 컬럼으로 인한 '조용한 실패'를 사전에 차단합니다.

        Args:
            cols: YAML에서 사용자가 지정한 컬럼명 리스트
            df: 실제 데이터 DataFrame
            context: 에러 메시지에 표시할 맥락 (예: "scaling", "encoding")

        Returns:
            검증 통과된 컬럼명 리스트 (입력과 동일)

        Raises:
            ValueError: 존재하지 않는 컬럼이 포함된 경우
        """
        missing = [c for c in cols if c not in df.columns]
        if missing:
            available = sorted(df.columns.tolist())
            raise ValueError(
                f"\n╔══════════════════════════════════════════════════════╗\n"
                f"║  ❌ [{context}] 존재하지 않는 컬럼이 지정되었습니다!  ║\n"
                f"╚══════════════════════════════════════════════════════╝\n"
                f"  지정된 컬럼 중 데이터에 없는 항목:\n"
                f"    → {missing}\n\n"
                f"  현재 데이터에 존재하는 컬럼 목록:\n"
                f"    → {available}\n\n"
                f"  💡 해결 방법:\n"
                f"    1) YAML의 {context} > cols 항목에서 오타를 확인하세요.\n"
                f"    2) 해당 컬럼이 drop_cols에 의해 제거되었는지 확인하세요.\n"
                f"    3) 모든 수치형/범주형 컬럼에 자동 적용하려면 cols: \"auto\"로 설정하세요.\n"
            )
        return cols

    def _resolve_scaler_cols(self, df: pd.DataFrame) -> list[str]:
        """스케일링 대상 컬럼을 결정합니다."""
        cols = self._scaling_cfg.get("cols", "auto")
        target_col = self._data_cfg.get("target_col", "target")
        if cols == "auto" or cols is None:
            return [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c != target_col]
        # 사용자가 직접 지정한 컬럼 → 검증 후 반환
        return self._validate_cols(list(cols), df, "scaling(스케일링)")

    def _resolve_encoder_cols(self, df: pd.DataFrame) -> list[str]:
        """인코딩 대상 컬럼을 결정합니다."""
        cols = self._encoding_cfg.get("cols", "auto")
        if cols == "auto" or cols is None:
            return [c for c in self._feat_cfg["cat_cols"] if c in df.columns]
        # 사용자가 직접 지정한 컬럼 → 검증 후 반환
        return self._validate_cols(list(cols), df, "encoding(인코딩)")

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "DataPreprocessor":
        """훈련 데이터의 통계량을 계산하고 sklearn 변환기를 학습합니다.
        
        저장되는 항목:
        - 결측치 처리용 통계량 (median/mean/mode)
        - sklearn Scaler 내부 파라미터 (mean_, scale_, data_min_, data_max_ 등)
        - sklearn Encoder 내부 파라미터 (categories_, mapping_ 등)
        
        Args:
            df: **훈련 데이터** DataFrame (타겟 컬럼 미포함 권장)
        
        Returns:
            self (메서드 체이닝 지원)
        """
        target_col = self._data_cfg.get("target_col", "target")
        cat_cols = self._feat_cfg["cat_cols"]

        # ── 1. 수치형 결측치 통계량 ──
        num_strategy = self._missing_cfg.get("numeric_strategy", "median")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

        self._num_fill_values = {}
        self._num_imputer = None
        self._numeric_cols_for_imputer = numeric_cols

        if num_strategy == "median":
            for col in numeric_cols:
                self._num_fill_values[col] = float(df[col].median())
        elif num_strategy == "mean":
            for col in numeric_cols:
                self._num_fill_values[col] = float(df[col].mean())
        elif num_strategy == "zero":
            for col in numeric_cols:
                self._num_fill_values[col] = 0.0
        elif num_strategy == "knn":
            from sklearn.impute import KNNImputer
            self._num_imputer = KNNImputer(n_neighbors=5)
            self._num_imputer.fit(df[numeric_cols])
        elif num_strategy == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            self._num_imputer = IterativeImputer(max_iter=10, random_state=self.config.get("project", {}).get("seed", 42))
            self._num_imputer.fit(df[numeric_cols])

        # ── 2. 범주형 결측치 통계량 ──
        cat_strategy = self._missing_cfg.get("categorical_strategy", "mode")
        fill_value = self._missing_cfg.get("fill_value", "MISSING")
        cat_cols_present = [c for c in cat_cols if c in df.columns]

        self._cat_fill_values = {}
        if cat_strategy == "mode":
            for col in cat_cols_present:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    self._cat_fill_values[col] = str(mode_val.iloc[0])
                else:
                    self._cat_fill_values[col] = fill_value
        elif cat_strategy == "constant":
            for col in cat_cols_present:
                self._cat_fill_values[col] = fill_value

        # (결측치를 먼저 채워야 sklearn 변환기 fit이 가능)
        df_filled = self._fill_missing(df)

        # ── 3. sklearn Scaler 학습 ──
        scaling_method = self._scaling_cfg.get("method", "none")
        if scaling_method != "none" and scaling_method in self._SCALER_MAP:
            self._scaler_cols = self._resolve_scaler_cols(df_filled)
            if self._scaler_cols:
                ScalerClass = self._import_class(self._SCALER_MAP[scaling_method])
                self._scaler = ScalerClass()
                self._scaler.fit(df_filled[self._scaler_cols])
                logger.info(
                    f"  Scaler({scaling_method}) fit 완료 — "
                    f"{len(self._scaler_cols)}개 컬럼, "
                    f"저장된 파라미터: {list(vars(self._scaler).keys())}"
                )

        # ── 4. sklearn Encoder 학습 ──
        encoding_method = self._encoding_cfg.get("method", "none")
        if encoding_method != "none" and encoding_method in self._ENCODER_MAP:
            self._encoder_cols = self._resolve_encoder_cols(df_filled)
            if self._encoder_cols:
                if encoding_method == "ordinal":
                    from sklearn.preprocessing import OrdinalEncoder
                    self._encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )
                    self._encoder.fit(df_filled[self._encoder_cols].astype(str))
                    logger.info(
                        f"  Encoder({encoding_method}) fit 완료 — "
                        f"{len(self._encoder_cols)}개 컬럼, "
                        f"categories 저장됨"
                    )
                elif encoding_method == "target":
                    try:
                        import category_encoders as ce
                        self._encoder = ce.TargetEncoder(cols=self._encoder_cols, smoothing=10)
                        
                        # 타겟값(y)이 누락된 경우 작동 불가
                        if y is None:
                            logger.warning("Target Encoding을 위해서는 fit()에 y값이 주어져야 합니다. 인코딩을 건너뜁니다.")
                            self._encoder = None
                        else:
                            self._encoder.fit(df_filled[self._encoder_cols].astype(str), y)
                            logger.info(
                                f"  Encoder({encoding_method}) fit 완료 — "
                                f"{len(self._encoder_cols)}개 컬럼"
                            )
                    except ImportError:
                        logger.warning("category_encoders 패키지가 설치되지 않아 Target Encoding을 건너뜁니다.")
                        self._encoder = None

        self._is_fitted = True
        logger.info(
            f"DataPreprocessor.fit() 완료 — "
            f"결측치: 수치형 {len(self._num_fill_values)}개, 범주형 {len(self._cat_fill_values)}개 / "
            f"Scaler: {scaling_method}({len(self._scaler_cols)}개) / "
            f"Encoder: {encoding_method}({len(self._encoder_cols)}개)"
        )
        return self

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치를 저장된 통계량으로 채웁니다 (내부 헬퍼)."""
        df = df.copy()
        num_strategy = self._missing_cfg.get("numeric_strategy", "median")
        cat_strategy = self._missing_cfg.get("categorical_strategy", "mode")
        cat_cols = self._feat_cfg["cat_cols"]

        if num_strategy == "drop":
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            df.dropna(subset=numeric_cols, inplace=True)
        elif self._num_imputer is not None:
            cols = [c for c in self._numeric_cols_for_imputer if c in df.columns]
            if cols:
                df[cols] = self._num_imputer.transform(df[cols])
        else:
            for col, fill_val in self._num_fill_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)

        if cat_strategy == "drop":
            cat_cols_present = [c for c in cat_cols if c in df.columns]
            df.dropna(subset=cat_cols_present, inplace=True)
        else:
            for col, fill_val in self._cat_fill_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)

        cat_cols_present = [c for c in cat_cols if c in df.columns]
        for col in cat_cols_present:
            df[col] = df[col].astype(str)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장된 훈련 데이터 통계량 + sklearn 변환기로 데이터를 변환합니다.
        
        변환 순서:
        1. 결측치 처리 (훈련 데이터의 median/mean/mode)
        2. sklearn Scaler 적용 (훈련 데이터의 mean_/scale_ 등)
        3. sklearn Encoder 적용 (훈련 데이터의 categories_)
        
        Args:
            df: 변환할 DataFrame (훈련/검증/테스트 모두 가능)
        
        Returns:
            변환된 DataFrame (원본 변경 없음)
        
        Raises:
            RuntimeError: fit()을 호출하지 않고 transform()을 호출한 경우
        """
        if not self._is_fitted:
            raise RuntimeError(
                "DataPreprocessor.fit()을 먼저 호출해야 합니다. "
                "데이터 누수를 방지하기 위해 훈련 데이터로 fit()한 뒤 transform()하세요."
            )

        # 1) 결측치 처리
        df = self._fill_missing(df)

        # 2) Scaler 적용
        if self._scaler is not None and self._scaler_cols:
            cols_present = [c for c in self._scaler_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._scaler.transform(df[cols_present])

        # 3) Encoder 적용
        if self._encoder is not None and self._encoder_cols:
            cols_present = [c for c in self._encoder_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._encoder.transform(df[cols_present].astype(str))

        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """fit()과 transform()을 연속으로 수행합니다.
        
        ⚠️ YAML의 enable_system_preprocessing이 false이면
           결측치/스케일링/인코딩을 모두 건너뛰고 원본을 그대로 반환합니다.
        
        Args:
            df: **훈련 데이터** DataFrame
            y: **타겟 데이터** Series (Target Encoding 사용 시 필수)
        
        Returns:
            변환된 DataFrame
        """
        enable_system = self._feat_cfg.get("enable_system_preprocessing", True)
        if not enable_system:
            logger.info("⏭️ 시스템 자동 전처리 건너뜀 (enable_system_preprocessing: false)")
            self._is_fitted = True
            return df
        return self.fit(df, y).transform(df)

    @property
    def fill_values(self) -> dict[str, dict]:
        """저장된 결측치 통계량을 반환합니다 (디버깅/확인용)."""
        return {
            "numeric": self._num_fill_values.copy(),
            "categorical": self._cat_fill_values.copy(),
        }

    @property
    def fitted_params(self) -> dict[str, dict]:
        """sklearn 변환기에 저장된 모든 학습 파라미터를 반환합니다 (디버깅/확인용).
        
        Returns:
            {
                "scaler": {"mean_": [...], "scale_": [...], ...},
                "encoder": {"categories_": [...], ...}
            }
        """
        params = {"scaler": {}, "encoder": {}}
        if self._scaler is not None:
            for k, v in vars(self._scaler).items():
                params["scaler"][k] = v
        if self._encoder is not None:
            for k, v in vars(self._encoder).items():
                params["encoder"][k] = v
        return params


def get_feature_names(config: dict, df: pd.DataFrame) -> list[str]:
    """설정을 기반으로 학습에 사용할 최종 피처 이름 목록을 반환합니다."""
    target_col = config.get("data", {}).get("target_col", "target")
    id_col = config.get("data", {}).get("id_col", None)
    drop_cols = config.get("features", {}).get("drop_cols", [])

    exclude_cols = [target_col] + drop_cols
    if id_col:
        exclude_cols.append(id_col)
    
    return [c for c in df.columns if c not in exclude_cols]


# ═══════════════════════════════════════════════════════════════
#  불균형 데이터 처리 (리샘플링)
# ═══════════════════════════════════════════════════════════════

# 지원하는 리샘플러 매핑
_RESAMPLER_MAP = {
    # ── 오버샘플링 (소수 클래스 복제/합성) ──
    "smote":            "imblearn.over_sampling.SMOTE",
    "adasyn":           "imblearn.over_sampling.ADASYN",
    "borderline_smote": "imblearn.over_sampling.BorderlineSMOTE",
    "random_over":      "imblearn.over_sampling.RandomOverSampler",
    # ── 언더샘플링 (다수 클래스 축소) ──
    "random_under":     "imblearn.under_sampling.RandomUnderSampler",
    "tomek":            "imblearn.under_sampling.TomekLinks",
    "enn":              "imblearn.under_sampling.EditedNearestNeighbours",
    # ── 복합 (오버 + 언더 결합) ──
    "smote_tomek":      "imblearn.combine.SMOTETomek",
    "smote_enn":        "imblearn.combine.SMOTEENN",
}


def _import_resampler(dotted_path: str):
    """문자열 경로에서 리샘플러 클래스를 동적 임포트합니다."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """훈련 데이터의 클래스 불균형을 해소하기 위해 리샘플링을 적용합니다.

    ⚠️ 반드시 훈련 데이터에만 적용하세요. 검증/테스트 데이터에 적용하면
       데이터 누수가 발생합니다.

    YAML 설정 예시::

        features:
          imbalance:
            method: "smote"              # 리샘플링 방법
            sampling_strategy: "auto"    # 복제 비율

    지원 방법:
        - none            : 사용 안 함
        - smote           : SMOTE (합성 소수 오버샘플링)
        - adasyn          : ADASYN (적응적 합성 오버샘플링)
        - borderline_smote: Borderline-SMOTE (경계 영역 집중)
        - random_over     : 랜덤 오버샘플링 (단순 복제)
        - random_under    : 랜덤 언더샘플링 (다수 클래스 축소)
        - tomek           : Tomek Links (경계 노이즈 제거)
        - enn             : ENN (편집 최근접 이웃 언더샘플링)
        - smote_tomek     : SMOTE + Tomek (합성 후 노이즈 제거)
        - smote_enn       : SMOTE + ENN (합성 후 이웃 정리)

    sampling_strategy 옵션:
        - "auto"   : 소수 클래스를 다수 클래스와 동일 수준으로 (기본)
        - "minority": 소수 클래스만 리샘플링
        - "not majority": 다수 클래스 제외 모두 리샘플링
        - 0.5      : 소수/다수 비율을 50%로 맞춤 (예: 소수 5000개, 다수 10000개)
        - 1.0      : 소수/다수 비율을 100%로 (= "auto"와 동일)
    """
    imbalance_cfg = config.get("features", {}).get("imbalance", {})
    method = imbalance_cfg.get("method", "none")

    if method == "none" or method is None:
        return X, y

    if method not in _RESAMPLER_MAP:
        available = list(_RESAMPLER_MAP.keys())
        raise ValueError(
            f"\n╔══════════════════════════════════════════════════════╗\n"
            f"║  ❌ 지원하지 않는 불균형 처리 방법입니다!             ║\n"
            f"╚══════════════════════════════════════════════════════╝\n"
            f"  입력값: '{method}'\n"
            f"  사용 가능한 방법: {available}\n"
        )

    seed = config.get("project", {}).get("seed", 42)
    sampling_strategy = imbalance_cfg.get("sampling_strategy", "auto")

    # 숫자형 문자열 → float 변환 (YAML에서 "0.5"로 입력될 수 있음)
    if isinstance(sampling_strategy, str):
        try:
            sampling_strategy = float(sampling_strategy)
        except ValueError:
            pass  # "auto", "minority" 등 문자열 그대로 사용

    try:
        ResamplerClass = _import_resampler(_RESAMPLER_MAP[method])

        # 리샘플러별 지원 파라미터가 다르므로 안전하게 구성
        kwargs = {}
        import inspect
        sig = inspect.signature(ResamplerClass.__init__)
        if "sampling_strategy" in sig.parameters:
            kwargs["sampling_strategy"] = sampling_strategy
        if "random_state" in sig.parameters:
            kwargs["random_state"] = seed

        resampler = ResamplerClass(**kwargs)

        logger.info(
            f"  [{method.upper()}] 훈련 데이터 리샘플링 적용 중... "
            f"(sampling_strategy={sampling_strategy})"
        )
        logger.info(f"  적용 전 클래스 분포:\n{y.value_counts().to_string()}")

        X_resampled, y_resampled = resampler.fit_resample(X, y)

        logger.info(
            f"  [{method.upper()}] 적용 후 데이터: {X_resampled.shape} "
            f"(기존 {X.shape})"
        )
        y_resampled = pd.Series(y_resampled, name=y.name)
        logger.info(f"  적용 후 클래스 분포:\n{y_resampled.value_counts().to_string()}")

        return X_resampled, y_resampled

    except ImportError:
        logger.warning(
            f"imblearn 패키지가 없어 {method}를 건너뜁니다. "
            f"(pip install imbalanced-learn)"
        )
        return X, y


# 하위 호환성: 기존 코드에서 apply_smote() 호출 시 동작 보장
apply_smote = apply_resampling


# ═══════════════════════════════════════════════════════════════
#  1. 데이터 로드
# ═══════════════════════════════════════════════════════════════
def load_raw_data(config: dict) -> pd.DataFrame:
    """원본 CSV 데이터를 로드합니다.
    
    Args:
        config: 전체 설정 dict
    
    Returns:
        원본 DataFrame
    """
    paths = get_paths(config)
    data_cfg = config.get("data", {})

    raw_path = paths.get("raw_data", "data/raw/train.csv")
    sep = data_cfg.get("separator", ",")
    encoding = data_cfg.get("encoding", "utf-8")

    logger.info(f"원본 데이터 로드: {raw_path}")
    df = pd.read_csv(raw_path, sep=sep, encoding=encoding)
    logger.info(f"  → shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  2. 안전한 전처리 (시스템 자동 + 사용자 커스텀)
# ═══════════════════════════════════════════════════════════════
#  ⭐ 사용자 정의 전처리는 src/custom_preprocessor.py 에서 수정하세요!
#     이 파일(preprocessor.py)은 시스템 자동 처리 전용입니다.
# ═══════════════════════════════════════════════════════════════
from src.custom_preprocessor import custom_safe_preprocess, custom_engineer_features


def safe_preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """데이터 누수가 없는 전처리를 수행합니다.
    
    실행 순서:
        1) 사용자 정의 안전 전처리 (custom_preprocessor.py)
        2) 시스템 자동 컬럼 제거 (YAML 설정 기반)
    
    Args:
        df: 원본 DataFrame
        config: 전체 설정 dict
    
    Returns:
        정리된 DataFrame
    """
    df = df.copy()
    feat_cfg = get_feature_config(config)

    # ── 1단계: 사용자 정의 안전 전처리 (custom_preprocessor.py) ──
    enable_custom = feat_cfg.get("enable_custom_preprocessing", True)
    if enable_custom:
        df = custom_safe_preprocess(df, config)
    else:
        logger.info("⏭️ 사용자 정의 전처리 건너뜀 (enable_custom_preprocessing: false)")

    # ── 2단계: 시스템 자동 컬럼 제거 (YAML의 drop_cols 기반) ──
    drop_cols = feat_cfg["drop_cols"]

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        logger.info(f"시스템 컬럼 제거: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    logger.info(f"안전한 전처리 완료 — shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  3. 피처 엔지니어링 (사용자 커스텀 → custom_preprocessor.py)
# ═══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """파생변수를 생성합니다.
    
    ⭐ 실제 파생변수 코딩은 src/custom_preprocessor.py 에서 수정하세요!
       이 함수는 custom_preprocessor의 함수를 호출하는 래퍼(wrapper)입니다.
    
    Args:
        df: 정제된 DataFrame
        config: 전체 설정 dict
    
    Returns:
        피처가 추가된 DataFrame
    """
    feat_cfg = get_feature_config(config)
    enable_custom = feat_cfg.get("enable_custom_preprocessing", True)
    if enable_custom:
        return custom_engineer_features(df, config)
    else:
        logger.info("⏭️ 사용자 정의 피처 엔지니어링 건너뜀 (enable_custom_preprocessing: false)")
        return df


# ═══════════════════════════════════════════════════════════════
#  4. 전체 전처리 파이프라인 (1단계: 누수 없는 작업만)
# ═══════════════════════════════════════════════════════════════
def preprocess_pipeline(config: dict) -> pd.DataFrame:
    """원본 데이터 로드 → 컬럼 정리 → 피처 생성 → parquet 저장.
    
    ⚠️ 결측치 처리(median/mean/mode)는 여기서 수행하지 않습니다.
    결측치 처리는 trainer.py의 K-Fold 루프 내부에서
    DataPreprocessor.fit(train).transform(val) 패턴으로 수행됩니다.
    
    Args:
        config: 전체 설정 dict
    
    Returns:
        전처리 완료된 DataFrame (결측치는 아직 남아있을 수 있음)
    """
    with Timer("전처리 파이프라인 (1단계)", logger):
        # 1) 원본 로드
        df = load_raw_data(config)

        # 2) 안전한 전처리 (컬럼 제거만)
        df = safe_preprocess(df, config)

        # 3) 피처 엔지니어링
        df = engineer_features(df, config)

        # 4) 저장
        paths = get_paths(config)
        processed_path = paths.get("processed_data", "data/processed/train_processed.parquet")
        save_dataframe(df, processed_path)
        logger.info(f"전처리 데이터 저장 완료: {processed_path}")

    return df


# ═══════════════════════════════════════════════════════════════
#  5. Hold-out 분할 (초기 훈련/검증 분리)
# ═══════════════════════════════════════════════════════════════
def split_holdout(
    df: pd.DataFrame,
    config: dict,
    holdout_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """데이터를 훈련/홀드아웃으로 분할하여 interim/ 에 저장합니다.
    
    ⚠️ 이 함수는 프로젝트 초기에 **딱 한 번만** 실행합니다.
    홀드아웃 데이터는 최종 평가 시(6단계)까지 절대 사용하지 않습니다.
    
    분류 문제: StratifiedShuffleSplit (클래스 비율 유지)
    회귀 문제: ShuffleSplit
    
    Args:
        df: 전처리 완료된 전체 DataFrame (타겟 포함)
        config: 전체 설정 dict
        holdout_ratio: 홀드아웃 비율 (기본 0.2 = 20%)
    
    Returns:
        (train_df, holdout_df) 튜플
    """
    from sklearn.model_selection import train_test_split

    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})
    paths = config.get("paths", {})
    
    target_col = data_cfg.get("target_col", "target")
    seed = project_cfg.get("seed", 42)
    task_type = project_cfg.get("task_type", "classification")

    stratify = df[target_col] if task_type == "classification" else None

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=stratify,
    )

    # 인덱스 리셋
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    # interim/ 에 저장
    interim_train = paths.get("interim_train", "data/interim/train.parquet")
    interim_holdout = paths.get("interim_holdout", "data/interim/holdout.parquet")
    
    save_dataframe(train_df, interim_train)
    save_dataframe(holdout_df, interim_holdout)

    logger.info(
        f"Hold-out 분할 완료 (ratio={holdout_ratio}) — "
        f"훈련: {train_df.shape}, 홀드아웃: {holdout_df.shape}"
    )
    logger.info(f"  훈련 저장: {interim_train}")
    logger.info(f"  홀드아웃 저장: {interim_holdout}")

    if task_type == "classification":
        logger.info(f"  훈련 타겟 분포:\n{train_df[target_col].value_counts().to_string()}")
        logger.info(f"  홀드아웃 타겟 분포:\n{holdout_df[target_col].value_counts().to_string()}")

    return train_df, holdout_df

