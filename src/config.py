"""
config.py — YAML 설정 로더
═══════════════════════════
configs/ 폴더의 YAML 파일을 읽어 Python dict로 변환하고,
경로 자동 생성, 피처·모델 설정 추출 등의 헬퍼 함수를 제공합니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils import ensure_dir


# ─── YAML 로더 ────────────────────────────────────────────────
def load_config(config_path: str) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환합니다.
    
    Args:
        config_path: YAML 파일 경로 (상대/절대 모두 가능)
    
    Returns:
        설정 딕셔너리
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        yaml.YAMLError: YAML 파싱 실패 시
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 경로 섹션이 있으면 디렉토리 자동 생성
    _ensure_output_dirs(config)
    return config


def _ensure_output_dirs(config: dict) -> None:
    """설정에 명시된 출력 디렉토리를 자동으로 생성합니다."""
    paths = config.get("paths", {})
    dir_keys = ["model_dir", "result_dir", "figure_dir", "submission_dir"]
    for key in dir_keys:
        dir_path = paths.get(key)
        if dir_path:
            ensure_dir(dir_path)


# ─── 피처 설정 추출 ───────────────────────────────────────────
def get_feature_config(config: dict) -> dict[str, Any]:
    """피처 엔지니어링 관련 설정을 추출합니다.
    
    Returns:
        {
            "drop_cols": [...],
            "cat_cols": [...],
            "num_cols": [...],
            "missing": {...},
        }
    """
    features = config.get("features", {})
    return {
        "drop_cols": features.get("drop_cols", []),
        "cat_cols": features.get("cat_cols", []),
        "num_cols": features.get("num_cols", []),
        "missing": features.get("missing", {}),
        "scaling": features.get("scaling", {}),
        "encoding": features.get("encoding", {}),
        # 전처리 On/Off 스위치 (STEP 4.5)
        "enable_custom_preprocessing": features.get("enable_custom_preprocessing", True),
        "enable_system_preprocessing": features.get("enable_system_preprocessing", True),
    }


# ─── 모델 파라미터 추출 ──────────────────────────────────────
def get_model_params(config: dict) -> dict[str, Any]:
    """모델 학습에 필요한 파라미터를 추출합니다.
    
    Returns:
        {
            "algorithm": str,
            "n_splits": int,
            "shuffle": bool,
            "early_stopping_rounds": int,
            "fixed_params": {...},
            "optuna": {...},
            "seed": int,
            "task_type": str,
        }
    """
    model_cfg = config.get("model", {})
    project_cfg = config.get("project", {})

    return {
        "algorithm": model_cfg.get("algorithm", "catboost"),
        "n_splits": model_cfg.get("n_splits", 5),
        "shuffle": model_cfg.get("shuffle", True),
        "early_stopping_rounds": model_cfg.get("early_stopping_rounds", 100),
        "fixed_params": model_cfg.get("fixed_params", {}),
        "optuna": model_cfg.get("optuna", {}),
        "seed": project_cfg.get("seed", 42),
        "task_type": project_cfg.get("task_type", "classification"),
    }


# ─── 경로 헬퍼 ────────────────────────────────────────────────
def get_paths(config: dict) -> dict[str, str]:
    """설정 파일의 paths 섹션을 반환합니다."""
    return config.get("paths", {})


def get_evaluation_config(config: dict) -> dict[str, Any]:
    """평가 관련 설정을 반환합니다."""
    return config.get("evaluation", {})


def get_visualization_config(config: dict) -> dict[str, Any]:
    """시각화 관련 설정을 반환합니다."""
    return config.get("visualization", {})
