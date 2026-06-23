스마트팜 작물 추천 AI — Phase 1 (ML)
======================================

과제명 : AI 기반 스마트팜 작물 추천 모델 개발 및 시각화
담당자 : 정재봉

[구성]
- notebooks/phase1_ml.ipynb    : 전체 분석 노트북
                                 (데이터 탐색 → EDA → 전처리 → 모델 3종 비교
                                  → 하이퍼파라미터 튜닝(GridSearchCV) → 모델 저장)
- app.py                       : Streamlit 데모 (슬라이더 입력 → 작물 추천 + 평가 시각화)
- models/phase1_crop_rf.pkl    : 학습·튜닝된 RandomForest 모델 (서빙용)
- data/Crop_recommendation.csv : 데이터셋 (Kaggle Crop Recommendation, 2,200행 × 8열)
- figures/phase1_ml/           : 모델 평가 그림 (정확도 비교·혼동행렬·피처 중요도)
- docs/                        : AI개발 수행내역서 (PDF)
- requirements.txt             : 의존 패키지

[실행 방법]
1) 패키지 설치
     pip install -r requirements.txt
2) 노트북 실행
     jupyter notebook notebooks/phase1_ml.ipynb
3) Streamlit 데모 실행
     streamlit run app.py

[모델 요약]
- 입력 7개(N·P·K·온도·습도·pH·강수량) → 작물 22종 분류
- 모델 3종 비교(LogisticRegression / RandomForest / XGBoost) 후 RandomForest 채택
- Test 정확도 99.5%, GridSearchCV 튜닝으로 과적합(Train-Test Gap) 완화 및 모델 경량화
