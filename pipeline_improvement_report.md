# 🔬 ML Pipeline Template: 구조 검토 및 개선 제안 보고서

지금까지 함께 뼈대를 세우고 고도화한 `ML Pipeline Template` 프로젝트의 전반적인 구조를 검토했습니다. 
현재 코드는 훌륭한 수준의 앙상블 기법(Stacking, EasyEnsemble), 데이터 누수 방지(Pipeline 래핑), 직관적인 YAML 제어(전처리 스위치)를 갖추고 있어 **실제 Kaggle 대회나 실무 프로젝트에 당장 투입하더라도 손색이 없는 상태**입니다.

하지만 "진짜 상용화(Production) 레벨" 또는 "더욱 완벽한 프레임워크"로 거듭나기 위해 몇 가지 **핵심적인 의문사항과 개선 제안**을 발견하여 보고드립니다.

---

## 💡 A. 아키텍처 및 파이프라인 흐름 개선 (우선순위: 높음)

### 1. 🚀 모델 서빙/추론용 `predict.py` (Inference Script) 부재
*   **현재 상태**: [run_pipeline.py](file:///c:/Workspace/05_ML/ML_Pipeline_Template/run_pipeline.py)는 데이터를 쪼개고(split), 모델을 튜닝/학습(dev)하며, 테스트셋을 평가(eval)하는 **"학습(Training)"** 기능에 집중되어 있습니다.
*   **문제점**: 학습이 끝난 최고 성능의 모델(`.pkl`)과 전처리기(`.pkl`)를 가지고, 내일 들어올 **"새로운 고객 데이터 1줄"**을 예측해주는 서비스용 스크립트가 명확하지 않습니다.
*   **개선 제안**: `src/predict.py` (또는 `api.py`)를 만들어, 저장된 전처리기 파이프라인과 모델을 한 번에 로드(Load)한 뒤, [raw_data](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/preprocessor.py#550-570) 1줄만 던져주면 바로 예측 확률을 뱉어내는 템플릿 코드를 추가하는 것을 강력히 추천합니다. (FastAPI 뼈대도 좋고, CLI 스크립트도 좋습니다)

### 2. ⚠️ 전처리기(Preprocessor) 저장 체계 일원화 확인 필요
*   **의문점**: 교차검증(K-Fold) 내에서는 데이터 누수를 막기 위해 Fold 단위로 [fit_transform](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/preprocessor.py#352-371)이 정상 작동하도록 잘 짰습니다. 그런데 100% 데이터를 다 쓴 최종 제출용 모델(평가용)을 만들 때, [DataPreprocessor](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/preprocessor.py#29-398) 클래스가 모델과 *함께 묶여서(Pipeline 객체로)* `model_dir`에 저장되고 있는지, 아니면 별도의 `.pkl`로 명확히 저장되는지 점검이 요망됩니다.
*   **개선 제안**: `joblib`이나 `pickle`을 이용할 때, [(전처리기 + 베이스 모델 + 메타 모델)](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/preprocessor.py#155-278)이 하나의 파이프라인 묶음으로 통합되어 저장되게 만들면 나중에 추론 시 에러 발생률이 0%에 가까워집니다.

---

## 💡 B. 머신러닝 성능 극대화 / OOV 대응 (우선순위: 중간)

### 3. 🛡️ 본 적 없는 카테고리 (OOV - Out of Vocabulary) 에러 방어
*   **현재 상태**: `Ordinal Encoding`이나 `Target Encoding`이 적용되어 있습니다. 
*   **문제점**: 학습 데이터(`train.csv`)의 `직업` 컬럼에는 "학생", "회사원"만 있었는데, 어느 날 새로운 데이터(`test.csv`)에 "프리랜서"가 들어오면 기존 인코더가 학습한 적 없는 단어라 치명적인 에러(ValueError)를 뱉고 파이프라인이 뻗어버릴 위험이 있습니다.
*   **개선 제안**: 
    1. 범주형 결측치 처리에 "알 수 없음(Unknown)"과 같은 고정 상수([fill_value](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/preprocessor.py#372-379))를 의도적으로 남기기.
    2. 인코더 파라미터에 `handle_unknown='ignore'` 또는 `handle_unknown='value'` 옵션을 명시적으로 주입하여, 모르는 단어는 최빈값이나 -1 로 안전하게 매핑되도록 확실히 다져두어야 합니다.

### 4. 🧠 CatBoost와 LightGBM의 "천연 범주형 기능" 충돌 방지
*   **현재 상태**: 시스템 전처리에서 사용자가 `encoding: ordinal` 등을 설정하면 데이터프레임 안의 문자열이 숫자로 싹 다 바뀝니다.
*   **문제점**: CatBoost나 LightGBM은 태생적으로 `cat_features` 인자만 넘겨주면, 자기들이 내부적으로 "문자열(String)이나 카테고리(Category) 타입"을 직접 지지고 볶아서 기가 막히게 성능을 뽑아냅니다. 그런데 우리가 미리 전처리기로 인코딩(명시적 변환)해버리면 오히려 이 핵심 강점을 잃어버리고 일반 트리 모델 수준의 성능으로 떨어집니다.
*   **개선 제안**: [trainer.py](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/trainer.py)에서 `algorithm == "catboost"` 또는 `"lightgbm"`일 경우, 내부 조건문을 달아서 **"범주형 인코딩 시스템 스위치"를 강제로 끄거나 우회(Bypass)** 시키고 원본 String(또는 Category 타입)을 그대로 모델 내부로 토스해주는 하드코딩 예외 처리를 넣으면 성능이 비약적으로 점프합니다.

---

## 💡 C. 관리와 시각화 (우선순위: 선택형)

### 5. 📊 실험 트래킹 (MLflow / W&B) 도입
*   **현재 상태**: 로컬 폴더(`results/`, `models/`) 경로를 YAML에 쓰고 그곳에 산출물이 쌓입니다.
*   **문제점**: YAML 설정값을 계속 바꿔가며 수십 번 실험하면, 전에 돌렸던 Random Forest 5번 실험과 이번 실험 중 어느 게 더 좋았는지 기록 찾기가 지옥이 됩니다. (나중에 덮어써지기도 함)
*   **개선 제안**: [evaluator.py](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/evaluator.py)나 [trainer.py](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/trainer.py) 최상단에 `mlflow`를 2~3줄 정도만 감싸주면(Decorator 형태 또는 Context Manager), 파라미터와 F1 스코어, 산출물 그래프가 웹 UI에 자동 기록됩니다. "Kaggle 그랜드마스터급 템플릿" 방점을 찍는 마무리가 될 수 있습니다.

### 6. 다중 분류(Multi-class)에 대비한 평가/시각화 유연성
*   **현재 상태**: ROC 커브, PR 커브 등 시각화 설정이 이진 분류(Binary)에 엄청나게 최적화되어 있습니다.
*   **개선 제안**: 타겟 클래스가 3개 이상일 경우(예: A, B, C), ROC 커브를 그릴 때 One-vs-Rest (OvR) 방식으로 Macro 평균 그래프를 뽑아내는 로직이 [evaluator.py](file:///c:/Workspace/05_ML/ML_Pipeline_Template/src/evaluator.py) 안에 안전하게 깔려 있는지(if-else 분기) 한 번 더 확인하면 아주 강력한 무기가 됩니다.

---

### 👨‍💻 다음 진행 추천 방향
위 6가지 제안 중, 현시점에서 가장 가성비가 좋고 제가 당장이라도 코드로 짜드릴 수 있는 베스트픽 3가지는 다음과 같습니다.

1. **CatBoost/LGBM 범주형 인코딩 자동 By-pass 처리 (성능 향상 보장)**
2. **Handle_Unknown 옵션을 도입한 OOV(본 적 없는 데이터) 폭파 방지 쉴드**
3. **새로운 데이터를 위한 `predict.py`(추론 API 예제) 파일 1개 추가**

이 중에서 가장 마음에 드시거나, 먼저 해결하고 싶은 개선 사항이 있으신가요? 명령만 내려주시면 즉시 구현에 돌입하겠습니다.
