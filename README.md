# 🌱 SmartFarm AI — ML 입문 (v1 · 노지 작물 추천)

> 📦 **이 레포는 ML 입문 단계(v1)입니다.** 스마트팜 특화 본편(잎 진단 DL · 처방 LLM · 다작물)은 → **[smartfarm_ai](https://github.com/luma200ok/smartfarm_ai)**
>
> 센서는 *환경 숫자*를 보여주지만, 이 AI는 *작물에 지금 뭘 해줘야 하는지*를 알려준다.
> Kaggle Crop Recommendation으로 **정형 ML(작물 추천)** 을 학습한 단계 — 범용 ML 입문 후, 본편에서 **국내 스마트팜 실데이터로 발전**한다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-라이브_데모-FF4B4B?logo=streamlit&logoColor=white)](https://smartfarm-ai.streamlit.app/)
![Phase 1](https://img.shields.io/badge/Phase_1_ML-완료-2E7D32)
![Accuracy](https://img.shields.io/badge/정확도-99.5%25-1565C0)

> 🔗 **GitHub:** https://github.com/luma200ok/smartfarm_ml_learn  
> 🚀 **라이브 데모(Phase 1):** https://smartfarm-ai.streamlit.app/

---

<a id="toc"></a>
## 📑 목차

1. [✨ 주요 기능](#features)
2. [🏗️ 아키텍처 — 어떻게 물려 도는가](#architecture)
3. [🔗 단계 현황](#status)
4. [① Phase 1 · ML — 작물 추천](#phase1) ✅ **완료**
5. [② Phase 2 · DL — 병해충 진단](#phase2) 🟡 계획
6. [③ Phase 3 · LLM — 자연어 처방](#phase3) 🟡 계획
7. [📂 프로젝트 구조](#structure)
8. [🛠️ Tech Stack](#tech)
9. [📑 문서 & 라이선스](#docs)

---

<a id="features"></a>
## ✨ 주요 기능

- 🔮 **작물 추천** — 토양·환경 7개 값 입력 → 적합 작물 22종 중 추천 + 신뢰도 Top3 + 추천 이유
- 🌾 **작물별 환경 가이드** — 작물 선택 → 적합 환경값(평균·최소·최대) 표
- 📊 **모델 평가·비교** — 3종 정확도 · 혼동행렬 · 피처 중요도 · VIF 시각화
- 📑 **자동 EDA 리포트** — `ydata-profiling` 리포트 임베드
- 🧠 **RandomForest 99.5%** — 교차검증(99.55%±0.25%)·GridSearchCV 튜닝 완료, 학습↔서빙 분리(`.pkl`)

---

<a id="architecture"></a>
## 🏗️ 아키텍처 — 어떻게 물려 도는가

```
Phase 1 (ML)  센서/토양 → 작물 추천 분류 ───┐
                                          ├→ models/ (공유)
Phase 2 (DL)  잎 사진 → 병 진단 (CNN) ─────┤
              환경 시퀀스 → 추세 (LSTM) ───┘
                                          ↓ 진단·예측 결과를 먹음
Phase 3 (LLM) DL진단 + 환경예측 + RAG → 자연어 처방 → 알림(텔레그램)
```

- **공유:** `data/`(원본) · `models/`(학습 산출물) · `src/common/`(유틸·한글 라벨)
- **분리:** `src/{ml,dl,llm}`(단계 코드) · 단계별 Streamlit 진입점 → 개별 배포

---

<a id="status"></a>
## 🔗 단계 현황

| Phase | 한 일 | 상태 | 평가 지표 |
|---|---|---|---|
| **① ML** | 정형 센서 → 작물 추천 분류 (sklearn) | ✅ **완료** | 정확도 **99.5%** (+F1·혼동행렬) |
| **② DL** | 잎 사진 병해충 진단 + 환경 시계열 (CNN·LSTM) | 🟡 계획 | 분류: 정확도·F1 / 시계열: RMSE·MAE |
| **③ LLM** | 진단+환경 → 자연어 처방 + 알림 (Claude·RAG) | 🟡 계획 | RAG 검색 정확도 + 정성 평가 |

> **평가 지표는 단계 성격에 따라 다름** — 분류(ML·DL)는 정확도, 시계열(LSTM)은 오차(RMSE), 자연어 생성(LLM)은 정성 평가. "정확도 99.5%"는 Phase 1에만 해당.

---

<a id="phase1"></a>
## ① Phase 1 · ML — 작물 추천 ✅

**정형 토양·환경 데이터만으로 작물 22종을 99.5% 정확도로 추천**하는 분류 모델 + Streamlit 웹 데모.

- **데이터:** Kaggle Crop Recommendation — 2,200행 × 8열 (작물 22종 × 100개, 완전 균형, 결측 0)
- **입력(X):** 토양 N·P·K · 온도 · 습도 · pH · 강수량 (7개) → **정답(y):** 적합 작물 22종

### 모델 3종 비교 (공정 비교 후 베스트 선정)

| 모델 | Accuracy | 비고 |
|---|---|---|
| LogisticRegression | 97.3% | 기준선(직선 분리) |
| **RandomForest** | **99.5%** | 🏆 **베스트 — 데모 채택 (GridSearchCV 튜닝)** |
| XGBoost | 99.3% | 강력하나 근소 패 |

> **교훈:** 최신·강력 모델(XGBoost)이 항상 1등은 아니다 — 데이터가 쉬우면 RandomForest로 이미 천장. **실제 비교를 통해서만** 알 수 있다.
> **핵심 피처:** 강수량·습도(물 관련)가 작물 가르기의 1·2위 — EDA 결론과 정확히 일치.

<p align="center">
  <img src="figures/phase1_ml/phase1_model_compare.png" width="48%" alt="모델 3종 정확도 비교" />
  <img src="figures/phase1_ml/phase1_rf_importance.png" width="48%" alt="RandomForest 피처 중요도" />
</p>
<p align="center"><sub>왼쪽 — 모델 3종 정확도(RF 99.5% 베스트) · 오른쪽 — 피처 중요도(강수량·습도 1·2위)</sub></p>

📄 **상세 수행내역서:** [docs/phase1_ml.md](docs/phase1_ml.md) — EDA·VIF·튜닝·평가 신뢰성 검증 전부 수록

<details>
<summary>🚀 <b>실행 방법</b></summary>

```bash
# 의존성 설치 (uv 권장)
uv venv && uv pip install -r requirements.txt

# Streamlit 데모 실행 — 저장된 모델(.pkl) 불러와 예측만 (재학습 X)
streamlit run app.py

# (선택) 자동 EDA 리포트 재생성 → reports/phase1_eda_profile.html
python src/ml/profile_report.py
```

> **학습 ↔ 서빙 분리:** 학습은 미리 1번(`notebooks/phase1_ml.ipynb` → `models/phase1_crop_rf.pkl`), 앱은 불러와 예측만(`transform`만, 재학습 없음).

</details>

<details>
<summary>📸 <b>데모 스크린샷 (탭 3종)</b></summary>

<p align="center">
  <img src="figures/phase1_ml/phase1_demo_tab1.png" width="80%" alt="예측 탭" />
  <br><sub>🔮 <b>탭1 · 예측하기</b> — 슬라이더로 환경값 입력 → 추천 작물 + 신뢰도 Top3</sub>
</p>
<p align="center">
  <img src="figures/phase1_ml/phase1_demo_tab2.png" width="80%" alt="작물별 환경 가이드 탭" />
  <br><sub>🌾 <b>탭2 · 작물별 환경 가이드</b> — 작물 선택 → 적합 환경값(평균·최소·최대)</sub>
</p>
<p align="center">
  <img src="figures/phase1_ml/phase1_demo_tab3.png" width="80%" alt="모델 평가·비교 탭" />
  <br><sub>📊 <b>탭3 · 모델 평가·비교</b> — 3종 정확도 비교 + 평가 시각화</sub>
</p>

</details>

---

<a id="phase2"></a>
## ② Phase 2 · DL — 병해충 진단 🟡 계획

> 잎 사진 → **CNN 병해충 진단** + 환경 시계열 **LSTM** 추세 예측. 정형 ML로 불가능한 새 능력(이미지·순서).

<details>
<summary>📋 <b>계획 요약</b></summary>

- **확정:** PyTorch · 작물 토마토 · 비전 데이터 PlantVillage
- **CNN:** 전이학습(ResNet/EfficientNet) → 질병 분류
- **LSTM/GRU:** 지난 N시간 환경 시퀀스 → 향후 추세
- **평가:** 분류 정확도·F1 / 시계열 RMSE·MAE

</details>

📄 상세 → [docs/phase2_dl.md](docs/phase2_dl.md)

---

<a id="phase3"></a>
## ③ Phase 3 · LLM — 자연어 처방 🟡 계획

> CNN은 라벨("탄저병 87%")만 뱉음 → 진단+환경 예측을 받아 **사람 말로 처방**하고 **알림**까지. (Claude API + RAG)

<details>
<summary>📋 <b>계획 요약</b></summary>

- **확정:** LLM Claude API · 알림 텔레그램 · RAG 농사로 재배가이드
- **입력:** Phase 2 CNN 진단 + Phase 1·2 환경 예측
- **방법:** function calling으로 처방 생성 + RAG(재배 가이드 검색) 결합
- **평가:** RAG 검색 정확도 + 정성 평가(사실성·유용성)

</details>

📄 상세 → [docs/phase3_llm.md](docs/phase3_llm.md)

---

<a id="structure"></a>
## 📂 프로젝트 구조

<details>
<summary>디렉터리 트리 보기</summary>

```
smartfarm_ai/
├─ README.md              ← (이 파일) 허브
├─ app.py                 ← Phase 1 Streamlit 데모 (단일 진입점)
├─ docs/                  ← PRD·로드맵 + 단계별 기록(phase1~3.md)·트러블슈팅
├─ data/                  ← 공유 데이터 (git 제외, Kaggle 재다운)
├─ models/                ← 공유 학습 모델 (.pkl — 앱이 로드)
├─ notebooks/             ← phase1_ml.ipynb (탐색~모델저장 end-to-end)
├─ reports/               ← 자동 EDA 리포트 (ydata-profiling HTML/JSON)
├─ figures/phase1_ml/     ← EDA·모델 평가 시각화
├─ src/{ml,dl,llm,common} ← 단계 코드 + 공유 유틸
│   └─ ml/                  preprocess · train_{logreg,rf,xgb} · tune_rf · profile_report …
└─ requirements.txt
```

</details>

---

<a id="tech"></a>
## 🛠️ Tech Stack

| 영역 | 사용 기술 |
|---|---|
| **ML (Phase 1)** | `scikit-learn` · `xgboost` · `pandas`/`numpy` · `ydata-profiling`(자동 EDA) |
| **시각화·데모** | `matplotlib`/`seaborn` · `Streamlit` |
| **DL (Phase 2, 계획)** | `PyTorch` (CNN 전이학습 · LSTM) · PlantVillage |
| **LLM (Phase 3, 계획)** | `Claude API` + RAG(농사로 재배가이드) · 텔레그램 알림 |

---

<a id="docs"></a>
## 📑 문서 & 라이선스

**문서**
- [PRD](docs/prd.md) — 제품 기획서 · [로드맵](docs/roadmap.md) — 단계별 기술 로드맵
- [데이터 출처](docs/data_sources.md) · [🔧 트러블슈팅](docs/troubleshooting.md) — 배포까지 막혔던 지점·해결

**라이선스**
- **코드:** [MIT License](LICENSE) © 2026 luma200ok(정재봉)
- **데이터:** [Kaggle Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) — 해당 페이지 표기를 따름 (학습·포트폴리오용)

<sub>[⬆ 목차로](#toc)</sub>
