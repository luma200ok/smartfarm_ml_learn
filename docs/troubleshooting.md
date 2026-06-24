# 🔧 트러블슈팅 기록 — SmartFarm AI

> 개발·배포 중 실제로 막혔던 지점과 해결 과정. **문제 → 원인 → 해결 → 교훈** 4단으로 기록.
> ← [README 허브](../README.md)

배포까지 가면서 "코드는 되는데 환경/직렬화/경로에서 깨지는" 전형적인 함정들을 정리했다. 라이브 데모: https://smartfarm-ai.streamlit.app/

---

## Phase 1 — ML

### T1. `ydata-profiling` import 실패 — `ModuleNotFoundError: No module named 'pkg_resources'`

| 구분 | 내용 |
|---|---|
| **문제** | 자동 EDA 리포트(`src/ml/profile_report.py`) 실행 시 `ydata-profiling` import에서 `pkg_resources` 모듈을 못 찾음 |
| **원인** | `ydata-profiling 4.18.x`가 내부적으로 `pkg_resources`(setuptools 제공)를 사용하는데, **setuptools 81+에서 `pkg_resources`가 제거**됨 → 최신 setuptools 환경에서 충돌 |
| **해결** | `requirements.txt`에 `setuptools<81` 핀 고정. pandas/numpy/sklearn 버전은 영향 없음을 확인 |
| **교훈** | 라이브러리 자체가 아니라 **전이 의존성(setuptools)의 버전 변화**가 깨뜨리는 경우 — 에러 메시지의 모듈명(`pkg_resources`)을 단서로 의존성 계보를 추적해야 한다 |

```text
# requirements.txt
setuptools<81   # ydata-profiling 4.18.x가 pkg_resources 사용 (setuptools 81+에서 제거됨)
```

---

### T2. 배포 환경에서 저장된 모델(`.pkl`) 로딩 호환성

| 구분 | 내용 |
|---|---|
| **문제** | 로컬에서 학습·저장한 `models/phase1_crop_rf.pkl`을 다른 환경(배포 서버)에서 불러올 때 버전 불일치 시 직렬화 비호환·경고 발생 위험 |
| **원인** | scikit-learn은 **학습 시점과 로딩 시점의 버전이 다르면** pickle 역직렬화가 깨지거나 `InconsistentVersionWarning`을 낸다 (내부 클래스 구조 변경) |
| **해결** | `requirements.txt`에 `scikit-learn==1.7.2`로 **학습·서빙 버전 고정**. 학습(노트북) ↔ 서빙(`app.py`)이 동일 버전을 쓰도록 보장 |
| **교훈** | "학습과 서빙은 다른 환경"이라는 ML 배포의 기본 함정. 모델을 파일로 넘길 땐 **프레임워크 버전도 함께 고정**해야 재현된다 |

---

### T3. Streamlit `MediaFileStorageError` — 평가 이미지 경로 불일치

| 구분 | 내용 |
|---|---|
| **문제** | Streamlit 앱의 「모델 평가」 탭에서 figure를 표시할 때 `MediaFileStorageError`로 이미지 로딩 실패 |
| **원인** | figures를 `figures/` → `figures/phase1_ml/` 하위로 정리하면서, 앱 코드가 참조하는 경로와 **실제 파일 위치가 어긋남** |
| **해결** | `app.py`에서 두 경로(`figures/` ↔ `figures/phase1_ml/`)를 **런타임에 자동 탐지**하도록 보정 (커밋 `61f9600`) |
| **교훈** | 폴더 구조 리팩터링 후엔 **하드코딩된 상대경로**가 따라오는지 점검. 배포 파일시스템은 로컬과 작업 디렉터리가 달라 경로 버그가 늦게 드러난다 |

```python
# app.py — 실제 있는 위치를 자동 선택
FIG = ROOT / "figures"
if not (FIG / "phase1_model_compare.png").exists() and (FIG / "phase1_ml" / "phase1_model_compare.png").exists():
    FIG = FIG / "phase1_ml"
```

---

### T4. macOS numpy 헛 경고 — `divide by zero` / `invalid value`

| 구분 | 내용 |
|---|---|
| **문제** | 노트북·스크립트 실행 중 numpy 연산에서 `divide by zero`, `invalid value encountered` 경고가 다수 출력 |
| **원인** | macOS의 **Accelerate BLAS 백엔드**가 내는 환경 의존적 헛 경고 — 실제 계산 결과나 코드 로직과는 무관 |
| **해결** | 해당 구간에서 `np.seterr(...)`로 경고 억제. 결과값(정확도·예측)이 동일함을 교차 확인 |
| **교훈** | 모든 경고가 버그는 아니다. **환경(BLAS) 유래 노이즈**와 **로직 버그**를 구분 — 결과 재현성으로 판별한다 |

---

## Phase 2 (DL) · Phase 3 (LLM)
> 진행 시 이 문서에 이어서 기록 예정.
