"""
tests/test_preprocessor.py — DataPreprocessor 단위 테스트
──────────────────────────────────────────────────────────
실행 방법:
    cd C:\\Workspace\\05_ML\\ML_Pipeline_Template
    .venv\\Scripts\\pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessor import DataPreprocessor


# ── 공통 픽스처 ──────────────────────────────────────────────────
@pytest.fixture
def base_config():
    """기본 테스트 설정 dict를 반환합니다."""
    return {
        "data": {"target_col": "target"},
        "project": {"seed": 42},
        "features": {
            "cat_cols": ["gender", "job_type"],
            "num_cols": ["age", "income"],
            "drop_cols": [],
            "enable_system_preprocessing": True,
            "missing": {
                "numeric_strategy": "median",
                "categorical_strategy": "mode",
                "fill_value": "MISSING",
            },
            "scaling": {"method": "none"},
            "encoding": {"method": "none"},
        },
    }


@pytest.fixture
def train_df():
    """학습용 샘플 데이터를 반환합니다."""
    return pd.DataFrame({
        "age": [25.0, 30.0, None, 45.0, 50.0],
        "income": [3000.0, 4000.0, 5000.0, None, 6000.0],
        "gender": ["M", "F", "M", None, "F"],
        "job_type": ["student", "worker", "student", "worker", "student"],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def val_df():
    """검증용 샘플 데이터를 반환합니다."""
    return pd.DataFrame({
        "age": [28.0, None, 55.0],
        "income": [3500.0, 4500.0, None],
        "gender": ["M", "freelancer", "F"],   # ← OOV 포함
        "job_type": ["student", "unknown_job", "worker"],  # ← OOV 포함
        "target": [0, 1, 0],
    })


# ── 1. 데이터 누수 방지 검증 ──────────────────────────────────────
class TestDataLeakage:
    """훈련 통계량이 아닌 데이터가 검증 단계로 유출되지 않는지 검증합니다."""

    def test_numeric_fillna_uses_train_statistics(self, base_config, train_df, val_df):
        """결측치를 훈련 데이터의 median으로 채우는지 검증합니다."""
        X_train = train_df.drop(columns=["target"])
        X_val = val_df.drop(columns=["target"])

        dp = DataPreprocessor(base_config)
        X_train_t = dp.fit_transform(X_train)
        X_val_t = dp.transform(X_val)

        # 훈련 데이터의 age median: sorted([25, 30, 45, 50]) → median = (30+45)/2 = 37.5
        train_age_median = train_df["age"].dropna().median()
        assert abs(dp._num_fill_values["age"] - train_age_median) < 1e-6, (
            "age 결측치 통계량이 훈련 데이터 median과 다릅니다."
        )

        # 검증 데이터의 결측 age는 훈련 median으로 채워져야 함
        val_missing_mask = val_df["age"].isna()
        assert not X_val_t[val_missing_mask]["age"].isna().any(), (
            "검증 데이터의 결측 age가 채워지지 않았습니다."
        )
        filled_vals = X_val_t[val_missing_mask]["age"].values
        assert all(abs(v - train_age_median) < 1e-6 for v in filled_vals), (
            "검증 데이터의 결측 age가 훈련 median 값으로 채워지지 않았습니다."
        )

    def test_fit_transform_vs_separate_calls(self, base_config, train_df):
        """fit_transform()과 fit() + transform() 결과가 동일한지 검증합니다."""
        X_train = train_df.drop(columns=["target"])

        dp1 = DataPreprocessor(base_config)
        result1 = dp1.fit_transform(X_train.copy())

        dp2 = DataPreprocessor(base_config)
        dp2.fit(X_train.copy())
        result2 = dp2.transform(X_train.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_without_fit_raises(self, base_config, val_df):
        """fit() 없이 transform() 호출 시 RuntimeError가 발생해야 합니다."""
        X_val = val_df.drop(columns=["target"])
        dp = DataPreprocessor(base_config)
        with pytest.raises(RuntimeError, match="fit"):
            dp.transform(X_val)


# ── 2. OOV(Out-of-Vocabulary) 에러 방어 검증 ─────────────────────
class TestOOVHandling:
    """학습 때 없던 카테고리가 들어와도 에러 없이 -1로 처리되는지 검증합니다."""

    def test_ordinal_encoder_oov_returns_minus_one(self, base_config, train_df, val_df):
        """OrdinalEncoder에서 OOV 카테고리가 -1로 매핑되는지 검증합니다."""
        config = dict(base_config)
        config["features"] = dict(base_config["features"])
        config["features"]["encoding"] = {"method": "ordinal"}

        X_train = train_df.drop(columns=["target"])
        X_val = val_df.drop(columns=["target"])  # "freelancer", "unknown_job" OOV 포함

        dp = DataPreprocessor(config)
        dp.fit_transform(X_train.copy())

        # 에러 없이 transform 완료되어야 함
        X_val_t = dp.transform(X_val.copy())

        # OOV 행에 -1이 부여되어야 함
        oov_row_idx = val_df.index[val_df["gender"] == "freelancer"].tolist()
        if oov_row_idx:
            assert X_val_t.loc[oov_row_idx[0], "gender"] == -1, (
                "OOV 카테고리 'freelancer'가 -1로 매핑되지 않았습니다."
            )

    def test_no_exception_on_oov_with_ordinal(self, base_config, train_df, val_df):
        """OOV 데이터가 포함되어도 예외가 발생하지 않아야 합니다."""
        config = dict(base_config)
        config["features"] = dict(base_config["features"])
        config["features"]["encoding"] = {"method": "ordinal"}

        X_train = train_df.drop(columns=["target"])
        X_val = val_df.drop(columns=["target"])

        dp = DataPreprocessor(config)
        dp.fit_transform(X_train.copy())

        # 예외 없이 실행되어야 함
        try:
            dp.transform(X_val.copy())
        except Exception as e:
            pytest.fail(f"OOV 데이터 처리 중 예외 발생: {e}")


# ── 3. 모델 빌더 반환 검증 ────────────────────────────────────────
class TestModelBuilder:
    """get_model()이 각 알고리즘별로 올바른 객체를 반환하는지 검증합니다."""

    @pytest.mark.parametrize("algorithm,expected_class_name", [
        ("randomforest", "RandomForestClassifier"),
        ("xgboost", "XGBClassifier"),
        ("logistic", "LogisticRegression"),
        ("decisiontree", "DecisionTreeClassifier"),
        ("knn", "KNeighborsClassifier"),
        ("naivebayes", "GaussianNB"),
    ])
    def test_get_model_returns_correct_type(self, algorithm, expected_class_name):
        """각 알고리즘 문자열에 대해 올바른 모델 클래스가 반환되는지 검증합니다."""
        from src.trainer import get_model
        model = get_model(algorithm, {}, "classification", 42)
        assert type(model).__name__ == expected_class_name, (
            f"알고리즘 '{algorithm}'의 반환 타입이 예상({expected_class_name})과 다릅니다: "
            f"{type(model).__name__}"
        )

    def test_get_model_unknown_raises(self):
        """지원하지 않는 알고리즘 문자열 입력 시 ValueError가 발생해야 합니다."""
        from src.trainer import get_model
        with pytest.raises(ValueError, match="지원하지 않는"):
            get_model("unknown_model_xyz", {}, "classification", 42)


# ── 4. system preprocessing 스위치 검증 ──────────────────────────
class TestPreprocessingSwitch:
    """enable_system_preprocessing: false 시 전처리가 건너뛰어지는지 검증합니다."""

    def test_skip_system_preprocessing(self, base_config, train_df):
        """enable_system_preprocessing이 false일 때 원본 데이터가 그대로 반환되는지 검증합니다."""
        config = dict(base_config)
        config["features"] = dict(base_config["features"])
        config["features"]["enable_system_preprocessing"] = False

        X_train = train_df.drop(columns=["target"])
        dp = DataPreprocessor(config)
        result = dp.fit_transform(X_train.copy())

        # 결측치가 그대로 남아 있어야 함 (처리 안 함)
        assert result["age"].isna().any() or result["income"].isna().any(), (
            "enable_system_preprocessing=false 시 결측치가 처리되면 안 됩니다."
        )
