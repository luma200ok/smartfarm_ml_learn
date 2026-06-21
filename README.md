# 🌱 SmartFarm AI — 멀티모달 재배 도우미 (ML → DL → LLM)

> 센서는 *환경 숫자*를 보여주지만, 이 AI는 *작물에 지금 뭘 해줘야 하는지*를 알려준다.
>
> 하나의 도메인(스마트팜)을 **정형(ML) → 이미지(DL) → 언어(LLM)** 멀티모달 3단으로 완주한 프로젝트.
> 세 단계는 한 레포 안에서 **톱니처럼 물려** 돌지만(데이터·모델 공유), **각 단계는 개별 배포**된다.

---

## 🔗 단계별 기록 & 라이브 데모

| Phase | 한 일 | 기록 | 라이브 데모 |
|---|---|---|---|
| **1 · ML** | 정형 센서 → 작물·환경 판단 (sklearn) | [📄 ml.md](docs/ml.md) | _(배포 후 링크)_ |
| **2 · DL** | 잎 사진 병해충 진단 + 환경 시계열 (CNN·LSTM) | [📄 dl.md](docs/dl.md) | _(배포 후 링크)_ |
| **3 · LLM** | 진단+환경 → 자연어 처방 + 알림 (RAG) | [📄 llm.md](docs/llm.md) | _(배포 후 링크)_ |

> 각 단계 .md = 문제 → 데이터 → 방법 → 결과 → 배운점 (포트폴리오 기록)

---

## 🏗️ 어떻게 물려 도는가

```
Phase 1 (ML)  센서/토양 → 작물·환경 분류 ─┐
                                          ├→ models/ (공유)
Phase 2 (DL)  잎 사진 → 병 진단 (CNN) ─────┤
              환경 시퀀스 → 추세 (LSTM) ───┘
                                          ↓ 진단·예측 결과를 먹음
Phase 3 (LLM) DL진단 + 환경예측 + RAG → 자연어 처방 → 알림
```

- **공유:** `data/`(원본), `models/`(학습 산출물), `src/common/`(유틸)
- **분리:** `src/ml·dl·llm`(단계 코드), `app/phaseN_*.py`(단계별 Streamlit 진입점 → 개별 배포)

---

## 📂 구조

```
smartfarm_ai/
├─ README.md              ← (이 파일) 허브
├─ docs/                  ← PRD·로드맵 + 단계별 기록(ml/dl/llm.md)
├─ data/                  ← 공유 데이터 (git 제외, Kaggle 재다운)
├─ models/                ← 공유 학습 모델 (dl 저장 → llm 로드)
├─ notebooks/             ← phase1_eda.ipynb, phase2_cnn.ipynb ...
├─ src/{ml,dl,llm,common} ← 단계 코드 + 공유 유틸
├─ app/                   ← phase1_ml.py · phase2_dl.py · phase3_llm.py (개별 배포)
└─ requirements.txt
```

---

## 📑 문서

- [PRD](docs/PRD_smartfarm.md) — 제품 기획서
- [ML→DL→LLM 로드맵](docs/ML_DL_LLM_roadmap.md) — 단계별 기술 로드맵
- [데이터 출처](docs/data_sources.md) — 단계별 데이터셋·URL 기록

## 🛠️ Tech Stack

`Python` · `scikit-learn`/`xgboost` (ML) · `PyTorch` (DL: CNN·LSTM) · `Claude API`+RAG (LLM) · `Streamlit` (배포)
