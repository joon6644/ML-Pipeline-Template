"""
utils.py — 범용 유틸리티 함수
═══════════════════════════════
시드 고정, 로깅, 파일 I/O, 디렉토리 생성, 소요시간 측정 등
파이프라인 전체에서 공통으로 쓰이는 기능을 모아둡니다.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# ─── 시드 고정 ─────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """실험 재현성을 위해 전역 시드를 고정합니다.
    
    Args:
        seed: 고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── 로깅 ──────────────────────────────────────────────────────
def get_logger(
    name: str = "pipeline",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """콘솔(+ 선택적 파일) 로거를 생성합니다.
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만)
        level: 로깅 레벨
    
    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 있다면 중복 추가 방지
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택)
    if log_file:
        ensure_dir(str(Path(log_file).parent))
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ─── 디렉토리 유틸리티 ─────────────────────────────────────────
def ensure_dir(dir_path: str | Path) -> Path:
    """디렉토리가 없으면 생성합니다.
    
    Args:
        dir_path: 생성할 디렉토리 경로
    
    Returns:
        생성(또는 이미 존재)한 Path 객체
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── DataFrame I/O ─────────────────────────────────────────────
def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """DataFrame을 파일로 저장합니다. 확장자에 따라 형식을 자동 결정합니다.
    
    지원 형식: .parquet, .csv
    
    Args:
        df: 저장할 DataFrame
        path: 저장 경로
    """
    path = Path(path)
    ensure_dir(path.parent)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix} (parquet, csv만 가능)")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """파일에서 DataFrame을 로드합니다. 확장자에 따라 형식을 자동 결정합니다.
    
    Args:
        path: 로드할 파일 경로
    
    Returns:
        pandas DataFrame
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix} (parquet, csv만 가능)")


# ─── 소요시간 측정 ─────────────────────────────────────────────
class Timer:
    """with 문으로 소요시간을 측정하는 컨텍스트 매니저.
    
    사용 예시::
    
        with Timer("학습") as t:
            model.fit(X, y)
        print(t.elapsed)   # 초 단위
    """

    def __init__(self, label: str = "", logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start
        msg = f"[{self.label}] 완료 — {self.elapsed:.2f}초 소요"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# ─── 기타 유틸리티 ─────────────────────────────────────────────
def format_number(value: float, decimals: int = 4) -> str:
    """숫자를 지정된 소수점 자리까지 포맷합니다."""
    return f"{value:.{decimals}f}"
