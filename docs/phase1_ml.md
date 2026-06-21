# 📄 Phase 1 — ML : 정형 센서 → 작물·환경 판단

> 단계별 기록 (문제 → 데이터 → 방법 → 결과 → 배운점) · 상태: 🟢 모델 ① 완료 (1-5)
> 데이터 확정: **Kaggle Crop Recommendation** · 학습 방식: 중간 단위로 끊어서(학습용)
> ← [README 허브](../README.md) · 다음 → [dl.md](dl.md)
>
> 🎯 **최종 산출물 = 「AI개발 수행내역서」 양식 보고서** (참고 샘플: 버섯 독·식용 예측모델 PDF, `~/Downloads/머신러닝 산출물_샘플2.pdf`).
> 우리 과제 = "버섯 2종 분류"를 "작물 22종 분류"로 치환한 동일 뼈대. 아래 섹션이 그 보고서 목차에 매핑됨.

---

## 1. 문제 (Problem)
- 스마트팜 대시보드는 현재 센서값만 표시 → "이 환경에 뭘 심어야/어떻게 해야" 판단은 사람 몫.
- 정형 데이터(온습도·토양·pH 등)로 **작물 적합도 / 환경 이상**을 자동 판단하고 싶다.

## 2. 데이터 (Data)
- **Kaggle Crop Recommendation Dataset** (N·P·K·온도·습도·pH·강수 → 작물)
- 확장: 스마트팜코리아 / 공공데이터포털 시설원예 환경 CSV
- _(다운로드 후 `data/`, EDA 요약 채우기)_

## 3. 방법 (Method)
- 분석 7단계 EDA (분포·상관 heatmap·결측/이상치)
- 모델: LogisticRegression → RandomForest → XGBoost (점진)
- 평가: train_test_split, Accuracy / F1 / 혼동행렬
- (선택) 환경 이상탐지 한 스푼

## 4. 결과 (Result)
### 모델별 정확도 비교
| 모델 | Accuracy | 비고 |
|---|---|---|
| **LogisticRegression** (1-5) | 97.3% | 첫 모델·기준선. 약점: rice·lentil·mothbeans |
| **RandomForest** (1-6) | **99.5%** | 오답 12→2개, rice recall 0.80→0.95. **🏆 베스트(데모 채택)** |
| **XGBoost** (1-7) | 99.3% | 강력 모델이나 데이터가 쉬워 RF에 근소 패 |

- **1-5 LogReg 상세:** 440개 중 428 정답. 오답 12개 핵심 = rice→jute 4개(둘 다 강수량 높음=환경 겹침), 나머지는 콩 사촌끼리. → 환경 닮은 작물끼리만 헷갈림 = 합리적.
- **1-6 RF 상세:** 440개 중 438 정답(99.5%). 비선형 경계로 rice↔jute 분리. **피처 중요도: 강수량 0.22·습도 0.22** 최상위(=1-3 EDA와 일치).
- **1-7 비교·선정:** RF 99.5% 🥇 / XGB 99.3% / LogReg 97.3% → **베스트 = RandomForest 확정**. 교훈: 최신 모델(XGB)이 항상 1등은 아님, 데이터가 쉬우면 RF로 천장. XGB 중요도는 인·칼륨 상위(모델마다 주목 피처 다름).
- 산출물: `figures/phase1_{logreg,rf,xgb}_confusion.png`, `phase1_{rf,xgb}_importance.png`, `phase1_model_compare.png`
- 🚀 배포(1-8): **Streamlit `app/phase1_ml.py` 로컬 동작 확인** — 슬라이더 7개 → 작물 추천 + 확률 Top3. 베스트 RF를 `models/phase1_crop_rf.pkl`(model+scaler+le)로 저장해 불러 씀.
- 🔗 라이브 데모: _(Streamlit Cloud 배포 시 URL — GitHub 푸시 후)_

## 5. 배운점 (Learned)
- ML 5단계 골격: 준비 → `fit`(학습) → `predict`(예측) → 평가 → 시각화. 모델만 갈아끼우면 흐름 동일.
- 평가는 3겹(Accuracy → classification_report → 혼동행렬). 한 숫자만 믿지 말 것.
- _(모델 비교·배포에서 막힌 점은 진행하며 추가)_

---

## 📋 부록 — 산출물 보고서 양식 매핑 (목표)
> 최종에 이 표대로 `phase1_ml.md` → 보고서로 정리. 샘플(버섯 PDF) 목차 ↔ 우리 진행 단계.

| 보고서 섹션 (수행내역서 양식) | 우리 산출물 | 상태 |
|---|---|---|
| 2.1 추진배경·목적 / 2.2 과제범위 | PRD·README | ✅ |
| 2.3 분석모델 구축 프로세스 (다이어그램) | _(보고서 작성 시)_ | ⬜ |
| 3.1 데이터 요약 | 1-2 `explore.py` | ✅ |
| 3.2 데이터 전처리 (인코딩·분리·스케일) | 1-4 `preprocess.py` | ✅ |
| 3.3 독립변수 선택 (상관 heatmap) | 1-3 `eda_plots.py` | ✅ |
| 4.1 모델 정의·최적화 (GridSearchCV 등) | 1-6+ RF·XGB | ⬜ |
| 4.2 모델학습·평가 (F1·혼동행렬·feature importance) | 1-5 `train_logreg.py` | ✅(LogReg분) |
| 5. 프로토타이핑 (Streamlit 화면) | `app/phase1_ml.py` | ⬜ |
| 6. 결론·제안 (확장 방향) | 최종 | ⬜ |
