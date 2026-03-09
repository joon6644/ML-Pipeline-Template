"""
custom_preprocessor.py — 사용자 정의 전처리 (프로젝트별 커스텀 영역)
═══════════════════════════════════════════════════════════════════
⭐ 이 파일은 프로젝트마다 사용자가 직접 수정하는 유일한 전처리 파일입니다.

실행 순서:
    1) 이 파일의 함수들이 먼저 적용됩니다 (사용자 정의 전처리)
    2) 그 다음 시스템 전처리(preprocessor.py의 DataPreprocessor)가
       YAML 설정에 따라 결측치 처리/스케일링/인코딩을 자동 수행합니다.

⚠️ 주의사항:
    - 여기에는 데이터 누수(Data Leakage)가 없는 작업만 넣으세요.
      (예: 컬럼 간 사칙연산, 로그변환, 범주 조합, 이상치 규칙 제거 등)
    - 훈련 데이터의 통계량(mean, std 등)을 사용하는 변환은
      시스템 전처리(preprocessor.py의 DataPreprocessor)에서 자동 처리됩니다.
    - 이 파일을 수정해도 시스템 전처리 로직에는 영향이 없으므로 안심하세요!
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  1. 사용자 정의 안전 전처리 (컬럼 정리, 이상치 제거 등)
# ═══════════════════════════════════════════════════════════════
def custom_safe_preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """데이터 누수가 없는 사용자 정의 전처리를 수행합니다.

    시스템(preprocessor.py)의 자동 전처리보다 **먼저** 실행됩니다.
    프로젝트의 도메인 지식을 활용하여 아래 영역에 코드를 추가하세요.

    사용 예시:
        - 특정 조건의 행 제거 (나이 > 200, 급여 < 0 등)
        - 타입 변환 (문자 → 숫자)
        - 컬럼명 클리닝

    Args:
        df: 원본 DataFrame (컬럼 제거 이전 상태)
        config: 전체 설정 dict (YAML에서 로드한 것)

    Returns:
        정리된 DataFrame
    """
    df = df.copy()

    # ────────────────────────────────────────────────────────
    #  💡 여기에 프로젝트별 안전 전처리를 추가하세요
    # ────────────────────────────────────────────────────────
    #
    # 예시 1) 명백한 이상치 제거
    # if "age" in df.columns:
    #     df = df[df["age"] < 200]
    #     logger.info("  → 나이 200 이상 이상치 제거 완료")
    #
    # 예시 2) 특정 컬럼 타입 강제 변환
    # if "zip_code" in df.columns:
    #     df["zip_code"] = df["zip_code"].astype(str)
    #
    # 예시 3) 컬럼명 소문자 통일
    # df.columns = [c.lower().strip() for c in df.columns]
    #
    # ────────────────────────────────────────────────────────

    logger.info(f"사용자 정의 안전 전처리 완료 — shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  2. 사용자 정의 피처 엔지니어링 (파생변수 생성)
# ═══════════════════════════════════════════════════════════════
def custom_engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """파생변수를 생성합니다.

    이 함수는 프로젝트에 맞게 사용자가 직접 코딩하는 핵심 영역입니다.
    아래 예시 코드를 참고하여 필요한 변환을 추가하세요.

    ⚠️ 주의: 여기에 추가하는 변환은 데이터 누수가 없는 작업만 넣으세요.
    (예: 컬럼 간 사칙연산, 로그변환, 범주 조합 등)
    훈련 데이터의 통계량(mean, std 등)을 사용하는 변환은
    DataPreprocessor 클래스(preprocessor.py)가 자동 처리합니다.

    Args:
        df: 정제된 DataFrame
        config: 전체 설정 dict

    Returns:
        피처가 추가된 DataFrame
    """
    df = df.copy()

    # ────────────────────────────────────────────────────────
    #  💡 여기에 프로젝트별 파생변수를 추가하세요
    # ────────────────────────────────────────────────────────
    #
    # 예시 1) 수치형 변수 비율 파생변수
    # if "income_total" in df.columns and "age" in df.columns:
    #     df["income_per_age"] = df["income_total"] / (df["age"] + 1)
    #
    # 예시 2) 범주형 조합 파생변수
    # if "gender" in df.columns and "income_type" in df.columns:
    #     df["gender_income"] = df["gender"] + "_" + df["income_type"]
    #
    # 예시 3) 로그 변환
    # if "income_total" in df.columns:
    #     df["log_income"] = np.log1p(df["income_total"])
    #
    # 예시 4) 날짜 파생변수
    # if "birth_date" in df.columns:
    #     df["birth_date"] = pd.to_datetime(df["birth_date"])
    #     df["birth_year"] = df["birth_date"].dt.year
    #     df["birth_month"] = df["birth_date"].dt.month
    #     df.drop(columns=["birth_date"], inplace=True)
    #
    # ────────────────────────────────────────────────────────

    logger.info(f"사용자 정의 피처 엔지니어링 완료 — 최종 shape: {df.shape}")
    return df
