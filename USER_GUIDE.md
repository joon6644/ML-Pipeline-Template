# 📖 ML Pipeline Template 사용자 가이드북

본 가이드북은 누구나 쉽게 머신러닝 파이프라인을 처음부터 끝까지 구축, 튜닝, 평가, 추론할 수 있도록 **명령어 기반으로 단계별 절차를 체계적으로 안내**하는 메뉴얼입니다.

---

## 📌 목차
0. [사전 준비 (Setup)](#0-사전-준비-setup)
1. [1단계: 원본 데이터 준비 (Data Preparation)](#1-1단계-원본-데이터-준비-data-preparation)
2. [2단계: 평가용 분할 (Hold-out Split)](#2-2단계-평가용-분할-hold-out-split)
3. [3단계: 베이스라인 빠른 확인 (Baseline)](#3-3단계-베이스라인-빠른-확인-baseline)
4. [4단계: 파라미터 튜닝 및 K-Fold 심층 평가 (Optuna)](#4-4단계-파라미터-튜닝-및-k-fold-심층-평가-optuna)
5. [5단계: 판단 기준치 튜닝 (Threshold Tuning)](#5-5단계-판단-기준치-튜닝-threshold-tuning)
6. [6단계: 최종 모델 학습 및 실전 평가 (Final Evaluation)](#6-6단계-최종-모델-학습-및-실전-평가-final-evaluation)
7. [7단계: 새로운 데이터 추론 (Prediction)](#7-7단계-새로운-데이터-추론-prediction)
8. [8단계: 실험 기록 및 비교 대시보드 (MLflow UI)](#8-8단계-실험-기록-및-비교-대시보드-mlflow-ui)
9. [9단계: 다중 모델 앙상블 (Ensemble)](#9-9단계-다중-모델-앙상블-ensemble)

---

## 0. 사전 준비 (Setup)

제공된 파일들을 구동하기 위한 파이썬 가상환경을 생성하고, 패키지들을 설치합니다.

```bash
# 1. 가상환경 생성 (.venv)
python -m venv .venv

# 2. 가상환경 활성화 (Windows 기준)
.venv\Scripts\activate

# (Mac / Linux인 경우)
# source .venv/bin/activate

# 3. 필수 패키지 설치
pip install -r requirements.txt
```

---

## 1. 1단계: 원본 데이터 준비 (Data Preparation)

가장 먼저 인공지능을 학습시킬 원본 데이터를 정해진 위치에 두어야 합니다.

1. 프로젝트 최상위 폴더 안에 `data/raw/` 폴더를 확인합니다. (없다면 생성)
2. 문제 해결에 쓸 학습용 CSV 파일(예: `train.csv`)을 `data/raw/` 안에 위치시킵니다.
   - 📂 **예시 파일 경로**: `data/raw/train.csv`

---

## 2. 2단계: 평가용 분할 (Hold-out Split)

최종적으로 **모델이 한 번도 보지 못한 데이터**에 대해 진짜 실력을 평가하기 위해, 데이터의 일부(약 10~20%)를 미리 떼어놓는 과정입니다.

```bash
# 전체 데이터의 20%를 향후 최종 평가(방어선) 용도로 분할 보관
python run_pipeline.py -c configs/catboost.yaml --split-holdout 0.2
```

> 💡 **참고**: 
> `-c configs/catboost.yaml` 부분을 돌리고자 하는 모델명으로 바꾸면 됩니다. (예: `configs/lgbm.yaml`, `configs/xgboost.yaml`)
> **결과물**: 이 단계를 거치면 `data/interim/train.parquet` 와 `data/interim/holdout.parquet`가 생성됩니다.

---

## 3. 3단계: 베이스라인 빠른 확인 (Baseline)

시간이 수십 분 걸릴 수 있는 하이퍼파라미터 자동 튜닝을 건너뛰고, 모델의 **기본(Default) 설정만으로 성능이 어느 정도 나오는지 가장 먼저 확인**하는 명령어입니다.

```bash
# 튜닝 과정을 생략하고 K-Fold 내부 테스트만 진행
python run_pipeline.py -c configs/catboost.yaml --skip-tuning
```

> 📊 **결과 확인**: 터미널에 찍히는 성능 점수와 `results/figures/`에 그려진 다양한 시각화 차트 (ROC 곡선, 피처 중요도 등)를 가볍게 살펴봅니다.

---

## 4. 4단계: 파라미터 튜닝 및 K-Fold 심층 평가 (Optuna)

파이프라인의 **핵심 메인 모드**입니다. AI가 50번 이상 값을 요리조리 바꿔가며 스스로 모델의 **최적 파라미터를 탐색(Optuna)** 하고, K-Fold 교차 검증을 통해 우연히 잘 나온 점수가 아닌 "안정적인 평균 점수"를 도출합니다.

```bash
# 기본 실행 (데이터 전처리 + 파라미터 튜닝 + 안정성 K-Fold 평가)
python run_pipeline.py -c configs/catboost.yaml

# (옵션) 만약 전처리는 이미 예전에 끝내서 튜닝만 빠르게 다시 해보고 싶을 때
python run_pipeline.py -c configs/catboost.yaml --skip-preprocess
```

> ⚙️ **튜닝 변경**: "몇 번 탐색할지", "어떤 지표(AUC, F1 등)를 끌어올릴지"는 모두 `.yaml` 설정 파일 내부에 직관적으로 적혀 있으니 메모장으로 열어 숫자를 수정하면 됩니다.

---

## 5. 5단계: 판단 기준치 튜닝 (Threshold Tuning)

*(※ 이진 분류 Task에만 해당)*  
불량품 탐지, 암 진단처럼 하나의 클래스가 매우 적은 **불균형 데이터**에서는, 무조건 확률 50%를 기준으로 자르는 것이 위험할 수 있습니다. 지표가 극대화되는 컷오프(기준 확률선)를 AI가 자동으로 찾아줍니다.

```bash
# F1-Macro 지표가 1순위로 최대화 되도록 최적의 임계치(Threshold)를 탐색
python run_pipeline.py -c configs/catboost.yaml --tune-threshold f1_macro
```

> 💾 결과: 찾아낸 최적의 기준값 소수점은 `models/best_threshold.pkl`에 자동 저장되며 나중에 새 데이터를 예측할 때 영구적으로 적용됩니다.

---

## 6. 6단계: 최종 모델 학습 및 실전 평가 (Final Evaluation)

위 4~5단계를 통해 확보한 **최적의 튜닝 파라미터**를 장착한 채로, 우리가 가진 **훈련 데이터 전체(100%)**를 다 부어서 단 하나의 "최종 모델"을 만듭니다. 그 후, **2단계에서 떼어두었던 Hold-out 데이터**를 불러와 첫 모의고사를 치릅니다.

```bash
# 전체 데이터로 최종 학습 후 Hold-out 평가 진행
python run_pipeline.py -c configs/catboost.yaml --final-eval

# (적극 권장) 덮어쓰기 방지를 위해 실행 ID(타임스탬프나 실험명)를 부여하여 저장
python run_pipeline.py -c configs/catboost.yaml --final-eval --run-id my_catboost_v1
```

> ✅ **최종 산출물 위치**: 
> - 🧠 모델 파일: `models/catboost_final.cbm` (인공지능 뇌), `models/data_preprocessor.pkl` (전처리 규칙)
> - 📊 성적표: `results/evaluation_report.txt`
> - 📈 시각화: `results/figures/my_catboost_v1/`

---

## 7. 7단계: 새로운 데이터 추론 (Prediction)

이제 현실의 새롭게 쏟아지는 데이터(정답칸이 비어있는 타겟 데이터)를 예측하고 실무에 적용(혹은 캐글 대회에 제출)할 시간입니다.

```bash
# 1. 가장 기본적인 예측 (결과물이 results/submissions 폴더 안에 생성됨)
python predict.py -c configs/catboost.yaml -i data/raw/test.csv

# 2. 결과가 반환될 위치나 이름을 직접 정해주고 싶을 때 (-o)
python predict.py -c configs/catboost.yaml -i data/raw/test.csv -o results/my_predict.csv

# 3. 임계값을 직접 지정하거나 (예: 0.45) 1/0 분류값이 아닌 45.1% 같은 확률값(Proba)으로 뽑고 싶을 때
python predict.py -c configs/catboost.yaml -i data/raw/test.csv --threshold 0.45 --proba
```

---

## 8. 8단계: 실험 기록 및 비교 대시보드 (MLflow UI)

"어제 돌렸던 설정이랑 오늘 돌린거랑 무슨 차이지?", "그래프 모양 비교해보고 싶다" 하실 때 사용합니다. 3~6단계를 거치며 만들어진 모든 데이터는 영구적으로 기록부(MLflow)에 자동 저장되고 있었습니다.

```bash
# 실험 기록 대시보드를 터미널에 띄우기
.venv\Scripts\mlflow ui --backend-store-uri file:./mlruns
```

1. 명령어를 치고 나면 브라우저(크롬 등)를 엽니다.
2. 주소창에 `http://localhost:5000` 을 치고 들어갑니다.
3. 역대급 성적이 나온 실험 체크, 지표 필터링, 하이퍼파라미터 비교, ROC 그래프 그림들을 **웹 UI 마우스 조작** 클릭만으로 깔끔하게 비교할 수 있습니다.

---

## 9. 9단계: 다중 모델 앙상블 (Ensemble)

강력한 단일 모델 하나로 벅차다면, 1~6단계를 각기 다른 모델(CatBoost, LightGBM, RandomForest 등)로 반복하여 최종 모델 여러 개를 모아보세요. **서로의 약점을 보완**하는 집단 지성이 가능합니다.

```bash
# 1. 서로 다른 3개 모델 결과를 모아 1:1:1로 공평하게 단순 평균내어 예측 투표
python -m src.ensemble --data data/raw/test.csv \
                       --models catboost_final.cbm xgboost_final.pkl lgbm_final.pkl

# 2. 그 중 특정 녀석이 성적이 훨씬 좋아서 가중치를 주고 싶을 때 (Cat: 50%, XGB: 30%, LGB: 20%)
python -m src.ensemble --data data/raw/test.csv \
                       --models catboost_final.cbm xgboost_final.pkl lgbm_final.pkl \
                       --weights 0.5 0.3 0.2 \
                       --output results/submissions/ensemble_w.csv
```
> 🎈 **장려점**: 앙상블을 돌려도 내부적으로 전처리가 똑같이 보장되며, 꼬일 걱정 없이 깔끔하게 새로운 결과물(`.csv`)만 툭 뱉어냅니다!
