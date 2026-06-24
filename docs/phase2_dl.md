# 📄 Phase 2 — DL : 잎 병해충 진단(CNN) + 환경 시계열(LSTM)

> 단계별 기록 (문제 → 데이터 → 방법 → 결과 → 배운점) · 상태: ⚪ 대기 (Phase 1 후)
> 확정: 프레임워크 **PyTorch** · 작물 **토마토** · 비전 데이터 **PlantVillage**
> ← 이전 [phase1_ml.md](phase1_ml.md) · [README 허브](../README.md) · 다음 → [phase3_llm.md](phase3_llm.md)

---

## 1. 문제 (Problem)
- **ML이 못 하는 일**을 추가한다 → ① 잎 사진으로 병 진단(이미지), ② 환경 흐름 예측(순서).
- 특히 **잎 병해충은 정형 ML로 불가능 → CNN 필수** (이 프로젝트 DL의 핵심 명분).

## 2. 데이터 (Data)
- **비전:** PlantVillage (Kaggle, 즉시) / AI Hub 농작물 병해충(한국)
- **시계열:** 스마트팜코리아 / 공공데이터 / 가상 시뮬레이션
- _(클래스 분포·이미지 수 채우기)_

## 3. 방법 (Method)
- **CNN:** 전이학습(ResNet/EfficientNet) → 질병 분류
- **LSTM/GRU:** 지난 N시간 환경 시퀀스 → 향후 추세
- 평가: 정확도·혼동행렬(비전), MAE(시계열), 학습곡선

## 4. 결과 (Result)
- _(비전 정확도, 시계열 오차, 예측 샘플/오분류 사례)_
- 모델 저장 → `models/` (Phase 3 LLM이 로드)
- 🚀 배포: Streamlit (`app/phase2_dl.py`) — 사진 업로드 → 진단
- 🔗 라이브 데모: _(URL)_

## 5. 배운점 (Learned)
- _(전이학습, 신경망 구조, "ML이 못 하던 걸 DL이 함" 체감, GPU/학습 이슈)_
