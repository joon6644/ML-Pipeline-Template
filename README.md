# 🌲 Tree-Based ML Pipeline Template

트리 모델(CatBoost, LightGBM, XGBoost, RandomForest) 기반의 **범용 머신러닝 파이프라인 템플릿**입니다.  
YAML 설정 파일 하나로 전처리 → 튜닝 → 학습 → 평가를 통제하며, **데이터 누수 방지**가 구조적으로 보장됩니다.

---

## 📁 프로젝트 구조

```
ML_Pipeline_Template/
│
├── data/
│   ├── raw/               # 원본 데이터 (train.csv)
│   ├── interim/            # 훈련/홀드아웃 분할 데이터 (parquet)
│   └── processed/          # 전처리 완료 데이터 (parquet)
│
├── configs/                # 모델별 YAML 설정 파일
│   ├── catboost.yaml
│   ├── lgbm.yaml
│   ├── randomforest.yaml
│   └── xgboost.yaml
│
├── src/                    # 핵심 파이프라인 모듈
│   ├── __init__.py
│   ├── config.py           # YAML 설정 로더
│   ├── utils.py            # 유틸리티 (시드, 로깅, I/O, Timer)
│   ├── preprocessor.py     # 전처리 + DataPreprocessor + Hold-out 분할
│   ├── trainer.py          # K-Fold 학습 · Optuna 튜닝 · 최종 학습
│   └── evaluator.py        # 지표 계산 · 시각화 · 리포트 생성
│
├── notebooks/              # EDA 및 실험용 노트북
├── models/                 # 학습된 모델 저장소
├── results/                # 평가 결과 · 시각화 · 리포트
│   ├── figures/
│   └── submissions/
│
├── run_pipeline.py         # 메인 실행 스크립트 (3가지 모드)
├── requirements.txt
└── .gitignore
```

---

## 🚀 실행 가이드 (7단계 워크플로우)

### 1단계: 원본 데이터 준비

```bash
# 원본 데이터를 data/raw/ 에 저장
cp train.csv data/raw/train.csv
```

### 2단계: EDA → Hold-out 분할

노트북에서 기초 EDA를 진행합니다 (중복 제거, 인덱스 제거 등).  
만족스러우면 **Hold-out 분할**을 실행합니다. 비율은 인자로 전달합니다.

```bash
# 20% 홀드아웃 분할 (검증 데이터의 비율)
python run_pipeline.py -c configs/catboost.yaml --split-holdout 0.2
```

> 📦 결과: `data/interim/train.parquet` + `data/interim/holdout.parquet`  
> ⛔ **홀드아웃은 최종 평가시까지 절대 사용하지 않습니다.**

### 3단계: 베이스라인 모델

```bash
# Optuna 없이 고정 파라미터로 빠른 베이스라인
python run_pipeline.py -c configs/catboost.yaml --skip-tuning
```

### 4단계: 전처리 함수 작성 (데이터 누수 방지)

`src/preprocessor.py`의 `engineer_features()` 함수에서 파생변수를 추가/제거합니다.

> ⚠️ **누수 없는 변환만** `engineer_features()`에 작성  
> (컬럼 간 사칙연산, 로그변환, 범주 조합 등)  
> 훈련 통계량이 필요한 변환(평균, 표준편차 등)은 `DataPreprocessor` 클래스에 추가

### 5단계: Optuna 튜닝 및 평가

```bash
# 기본 모드: K-Fold 평가 + Optuna 튜닝
python run_pipeline.py -c configs/catboost.yaml

# 임계값 수동 튜닝 (이진 분류 전용, f1_macro 기준)
python run_pipeline.py -c configs/catboost.yaml --tune-threshold f1_macro

# 전처리 건너뛰고 튜닝만 재실행
python run_pipeline.py -c configs/catboost.yaml --skip-preprocess

# 다른 모델로 실험 (각 YAML에 GPU 기본 활성화 설정됨)
python run_pipeline.py -c configs/lgbm.yaml
python run_pipeline.py -c configs/xgboost.yaml
python run_pipeline.py -c configs/randomforest.yaml
```

**💡 다양한 K-Fold 분할 지원**
YAML의 `model.split_strategy`를 변경하여 데이터 특성에 맞는 교차 검증을 사용할 수 있습니다.
- `stratified`: 클래스 비율 유지 (분류 기본)
- `kfold`: 단순 무작위 분할 (회귀 기본)
- `group`: 그룹(고객 ID 등)을 기준으로 분할하여 누수 방지 (`group_col` 지정 필수)
- `timeseries`: 시간에 따른 데이터 절단 (과거로 미래 예측)

### 6단계: 반복 개선

4~5단계를 반복합니다. `results/figures/`의 시각화와 `results/evaluation_report.txt`를 참고합니다.

### 7단계: 최종 평가

```bash
# 전체 훈련 데이터로 최종 모델 학습 → 홀드아웃 평가 1회
python run_pipeline.py -c configs/catboost.yaml --final-eval
```

> 📊 결과: `results/evaluation_report.txt`, `results/figures/*`  
> 🧠 모델: `models/{algorithm}_final.*`

---

## 📊 지원하는 평가 지표

### 분류 (Classification)

| 지표 | 설명 |
|------|------|
| Accuracy | 정확도 |
| Precision (Macro/Micro) | 정밀도 |
| Recall (Macro/Micro) | 재현율 |
| F1 Score (Macro/Micro/Weighted) | 조화평균 |
| ROC-AUC | ROC 곡선 하 면적 |
| PR-AUC (Average Precision) | PR 곡선 하 면적 |
| Log Loss | 로그 손실 |
| Brier Score | 예측 확률 교정(Calibration) 오차 (낮을수록 좋음) |
| Cohen's Kappa | 무작위 예측을 보정한 추정 신뢰도 (불균형 데이터 적합) |

### 회귀 (Regression)

| 지표 | 설명 |
|------|------|
| RMSE | 평균 제곱근 오차 |
| MAE | 평균 절대 오차 |
| R² | 결정 계수 |

---

## 📈 지원하는 시각화

| 차트 | YAML 설정 키 |
|------|-------------|
| Confusion Matrix | `confusion_matrix: true` |
| Feature Importance (Bar) | `feature_importance: true` |
| ROC Curve | `roc_curve: true` |
| Precision-Recall Curve | `pr_curve: true` |
| Calibration Curve | `calibration_curve: true` |
| Prediction Scatter (확률 산점도) | `prediction_scatter: true` |
| Lift / Gain Curve | `lift_gain_curve: true` |
| SHAP Summary (Beeswarm) | `shap_summary: true` |
| SHAP Waterfall (단일 샘플 기여도) | `shap_waterfall: true` |
| SHAP Dependence (피처별 산점도) | `shap_dependence: true` |
| PDP (Partial Dependence Plot) | `partial_dependence: true` |

---

## 🛡️ 데이터 누수 방지

이 파이프라인은 **구조적으로** 데이터 누수를 방지합니다:

```
K-Fold 루프 내부:
  1) DataPreprocessor.fit(X_train)     ← 훈련 데이터만으로 통계량 학습
  2) X_train = dp.transform(X_train)
  3) X_val   = dp.transform(X_val)     ← 훈련 통계량으로 검증 처리
  4) model.fit(X_train, y_train)
```

- 결측치 처리(median, mean, mode)는 **K-Fold 루프 내부에서** 매 Fold마다 훈련 데이터 기준으로 수행
- 고급 대치법인 `knn`, `iterative` 회귀대치 옵션 추가
- 클래스 불균형이 극심한 경우 훈련 데이터에만 오버샘플링을 적용하는 `method: "smote"` 옵션 내장
- 홀드아웃 데이터는 `--final-eval` 시에만 활용

---

## 🧪 MLflow 로컬 실험 추적
이 파이프라인은 `mlflow` 패키지가 설치되어 있다면, 튜닝이 마무리된 후 결과를 **자동으로 로컬 저장소에 기록**합니다.
- YAML 내 `project.mlflow_uri` (기본: `file:./mlruns`), `project.name`으로 실험 공간을 자동 생성
- Optuna에 의한 최적 하이퍼 파라미터, 평가 지표들을 스냅샷으로 영구 저장해 언제든 웹 UI(`mlflow ui`)로 비교 분석 가능

---

## ⚙️ YAML 설정 구조

각 알고리즘별 YAML 파일은 동일한 구조를 따릅니다:

```yaml
project:    # 프로젝트명, 시드, 태스크 유형
paths:      # 데이터/모델/결과 저장 경로
data:       # 타겟 컬럼, ID 컬럼, 구분자
features:   # 제거/범주형/수치형 설정, 결측치 처리, 스케일링(scaling), 인코딩(encoding)
model:      # 알고리즘, K-Fold 수, 분할 전략(split_strategy), GPU(device), 고정 파라미터, Optuna 범위
evaluation: # 최적화 지표, 임계값 범위
visualization:  # 시각화 On/Off 설정
```

---

## 🔧 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# 패키지 설치
pip install -r requirements.txt
```

### 주요 의존 패키지

| 패키지 | 용도 |
|--------|------|
| `pandas`, `numpy` | 데이터 처리 |
| `category_encoders` | 타겟 범주형 인코딩 (Target Encoding) |
| `scikit-learn` | 전처리, 평가 지표, RandomForest |
| `catboost` | CatBoost 모델 |
| `lightgbm` | LightGBM 모델 |
| `xgboost` | XGBoost 모델 |
| `optuna` | 하이퍼파라미터 자동 튜닝 (OOM 등 에러 발생 시 해당 트라이얼 건너뛰기 기능 포함) |
| `shap` | 모델 해석 (SHAP) |
| `matplotlib`, `seaborn` | 시각화 |
| `mlflow` | (선택) 로컬 실험 추적 (파라미터/지표 자동저장) |
| `imbalanced-learn` | (선택) SMOTE 오버샘플링 지원 |

---

## 📝 지원 모델

| 알고리즘 | 설정 파일 | `model.algorithm` |
|----------|-----------|-------------------|
| CatBoost | `configs/catboost.yaml` | `catboost` |
| LightGBM | `configs/lgbm.yaml` | `lightgbm` |
| XGBoost | `configs/xgboost.yaml` | `xgboost` |
| RandomForest | `configs/randomforest.yaml` | `randomforest` |
| Balanced Random Forest | `configs/balanced_rf.yaml` | `balancedrandomforest` |

---

## 🏆 다중 모델 앙상블 (Ensemble & Blending)

각 YAML 설정 파일 별로 모델들을 학습(`--final-eval`) 시켜 저장했다면, `ensemble.py` 스크립트를 통해 손쉽게 하나의 결과로 합쳐 성능을 극대화할 수 있습니다.

```bash
# 기본 사용법 (가중치 1:1 평균)
python -m src.ensemble --data data/raw/test.csv \\
                       --models catboost_final.cbm xgboost_final.pkl lgbm_final.pkl

# 특정 모델에 가중치 크게 분배 (Cat: 50%, XGB: 30%, LGBM: 20%)
python -m src.ensemble --data data/raw/test.csv \\
                       --models catboost_final.cbm xgboost_final.pkl lgbm_final.pkl \\
                       --weights 0.5 0.3 0.2 \\
                       --output results/submissions/my_ensemble_sub.csv
```
* **주의**: 앙상블 스크립트는 내부적으로 `--dp` 플래그를 통해 `models/data_preprocessor.pkl`를 로드하여 새로운 CSV 파일에 파이프라인의 훈련 통계량(결측치, 스케일링, 인코딩)을 동일하게 자동 변환시켜 적용합니다.
