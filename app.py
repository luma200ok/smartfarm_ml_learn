"""
스마트팜 작물 추천 — Streamlit 데모 (제출용 app.py)

흐름: 슬라이더 입력 → 저장된 RandomForest 묶음(.pkl) 불러옴 → 예측 → 추천 작물 표시
실행: streamlit run app.py

★ 학습 안 함! notebooks/phase1_ml.ipynb 에서 저장한 모델을 '불러와' 예측만 한다.
※ 제출용 단일 파일 — 영↔한 매핑을 인라인해 src/ 폴더 없이도 동작.
   필요 동반 파일: models/phase1_crop_rf.pkl, figures/phase1_*.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phase1_crop_rf.pkl"
# 모델 평가 그림 폴더 — figures/ 또는 figures/phase1_ml/ 중 실제 있는 곳을 자동 선택
FIG = ROOT / "figures"
if not (FIG / "phase1_model_compare.png").exists() and (FIG / "phase1_ml" / "phase1_model_compare.png").exists():
    FIG = FIG / "phase1_ml"
REPORT_PATH = ROOT / "reports" / "phase1_eda_profile.html"   # YData Profiling 자동 EDA 리포트

# ── 영어 → 한글 표시 매핑 (화면 표시 전용, 모델 내부는 영어 유지) ──
CROP_KO = {
    "rice": "벼(쌀)", "maize": "옥수수", "chickpea": "병아리콩", "kidneybeans": "강낭콩",
    "pigeonpeas": "비둘기콩", "mothbeans": "모스빈", "mungbean": "녹두", "blackgram": "검은녹두",
    "lentil": "렌틸콩", "pomegranate": "석류", "banana": "바나나", "mango": "망고",
    "grapes": "포도", "watermelon": "수박", "muskmelon": "머스크멜론", "apple": "사과",
    "orange": "오렌지", "papaya": "파파야", "coconut": "코코넛", "cotton": "목화",
    "jute": "황마", "coffee": "커피",
}
FEATURE_KO = {
    "N": "질소(N)", "P": "인(P)", "K": "칼륨(K)", "temperature": "온도(℃)",
    "humidity": "습도(%)", "ph": "산도(pH)", "rainfall": "강수량(mm)",
}
ko_crop = lambda c: CROP_KO.get(c, c)

# 슬라이더 범위 (데이터 min~max 기반) : (최소, 최대, 기본값)
RANGES = {
    "N":           (0,   140, 50),
    "P":           (5,   145, 50),
    "K":           (5,   205, 50),
    "temperature": (8.0,  44.0, 25.0),
    "humidity":    (14.0, 100.0, 70.0),
    "ph":          (3.5,  10.0, 6.5),
    "rainfall":    (20.0, 300.0, 100.0),
}


# @st.cache_resource = 모델을 매번이 아니라 '한 번만' 불러와 재사용 (속도)
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def predict_ui(model, scaler, le, prof_mean):
    """탭1 — 환경값 입력 → 작물 추천 (+ 추천 이유)"""
    st.subheader("환경값 입력")
    cols = st.columns(2)
    values = []
    for i, (key, (lo, hi, default)) in enumerate(RANGES.items()):
        with cols[i % 2]:
            v = st.slider(FEATURE_KO[key], float(lo), float(hi), float(default))
        values.append(v)

    if st.button("작물 추천 받기", type="primary"):
        # 입력값 → 학습 때와 같은 잣대로 변환 (scaler) → 모델 예측
        # 컬럼 이름 붙여 DataFrame으로 (scaler가 이름 기준으로 학습돼서 경고 방지)
        X = pd.DataFrame([values], columns=list(RANGES.keys()))
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]          # 작물 번호
        crop_en = le.inverse_transform([pred])[0]  # 번호 → 영어 이름

        st.success(f"### 🌾 추천 작물 : {ko_crop(crop_en)} ({crop_en})")

        # 보너스: 확률 top 3 (모델이 얼마나 확신하나)
        proba = model.predict_proba(X_scaled)[0]
        top3 = np.argsort(proba)[::-1][:3]
        st.subheader("추천 신뢰도 Top 3")
        for idx in top3:
            name_en = le.classes_[idx]
            st.write(f"- {ko_crop(name_en)} ({name_en}) : {proba[idx]:.1%}")

        # 추천 이유: 내 입력 vs 추천 작물 '전형값' 비교
        st.subheader(f"💡 왜 {ko_crop(crop_en)}? — 내 입력 vs {ko_crop(crop_en)} 평균 환경")
        typical = prof_mean.loc[crop_en]
        cmp = pd.DataFrame(
            {"내 입력": values,
             f"{ko_crop(crop_en)} 평균": [typical[k] for k in RANGES.keys()]},
            index=[FEATURE_KO[k] for k in RANGES.keys()],
        )
        st.table(cmp)
        st.caption("두 값이 비슷할수록 그 작물에 알맞은 환경입니다.")


def guide_ui(le, prof_mean, prof_min, prof_max):
    """탭2 — 작물별 적합 환경 가이드 (작물 → 환경값)"""
    st.subheader("🌾 작물별 적합 환경 가이드")
    st.caption("작물을 고르면 그 작물이 잘 자라는 환경값(평균·범위)을 보여줍니다.")

    ko2en = {ko_crop(en): en for en in le.classes_}
    sel_ko = st.selectbox("작물 선택", sorted(ko2en.keys()))
    en = ko2en[sel_ko]

    tbl = pd.DataFrame(
        {"평균": [prof_mean.loc[en, k] for k in RANGES.keys()],
         "최소": [prof_min.loc[en, k] for k in RANGES.keys()],
         "최대": [prof_max.loc[en, k] for k in RANGES.keys()]},
        index=[FEATURE_KO[k] for k in RANGES.keys()],
    )
    st.table(tbl)
    st.caption(f"예: {sel_ko}는 강수량 평균 {prof_mean.loc[en, 'rainfall']:.0f}mm, "
               f"습도 {prof_mean.loc[en, 'humidity']:.0f}% 환경을 좋아합니다.")


def eval_ui():
    """탭3 — 모델 학습 결과 (저장된 평가 그림 불러와 표시)"""
    st.subheader("세 모델 정확도 비교")
    st.table(pd.DataFrame({
        "모델": ["RandomForest 🏆", "XGBoost", "LogisticRegression"],
        "정확도": ["99.5%", "99.3%", "97.3%"],
        "비고": ["베스트(데모 채택)", "강력하나 근소 패", "기준선"],
    }))
    st.image(str(FIG / "phase1_model_compare.png"), caption="모델별 정확도 비교")

    st.subheader("베스트 모델(RandomForest) 상세")
    st.image(str(FIG / "phase1_rf_confusion.png"),
             caption="혼동행렬 — 대각선=정답, 밖=헷갈린 작물")
    st.image(str(FIG / "phase1_rf_importance.png"),
             caption="피처 중요도 — 강수량·습도가 작물 가르기에 핵심")


def eda_report_ui():
    """탭4 — YData Profiling 자동 EDA 리포트 (HTML 임베드)"""
    st.subheader("📑 자동 EDA 리포트 (YData Profiling)")
    st.caption("ydata-profiling으로 생성한 자동 EDA 리포트입니다. (요약통계·분포·상관·결측·품질경고)")
    if REPORT_PATH.exists():
        import streamlit.components.v1 as components
        components.html(REPORT_PATH.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.warning("리포트 파일이 없습니다. 아래 명령으로 먼저 생성하세요.")
        st.code("python src/ml/profile_report.py", language="bash")


def main():
    st.title("🌱 스마트팜 작물 추천")
    st.caption("토양·환경 값으로 적합 작물 추천 + 모델 평가 (RandomForest 99.5%)")

    # 탭이 너무 작아서 안 보임 → 크고 굵게 + 선택 탭 강조 (CSS 주입)
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 12px 28px;
        background-color: #F1F8E9;
        color: #2E5A1C;                 /* 연두 배경에 진초록 글자 (다크 테마서도 보이게 고정) */
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4C9A2A;
        color: white !important;        /* 선택 탭: 진초록 배경 + 흰 글자 */
    }
    </style>
    """, unsafe_allow_html=True)

    bundle = load_bundle()
    model, scaler, le = bundle["model"], bundle["scaler"], bundle["le"]
    prof_mean, prof_min, prof_max = bundle["prof_mean"], bundle["prof_min"], bundle["prof_max"]

    tab_predict, tab_guide, tab_eval, tab_report = st.tabs(
        ["🔮 예측하기", "🌾 작물별 환경 가이드", "📊 모델 평가·비교", "📑 자동 EDA 리포트"])
    with tab_predict:
        predict_ui(model, scaler, le, prof_mean)
    with tab_guide:
        guide_ui(le, prof_mean, prof_min, prof_max)
    with tab_eval:
        eval_ui()
    with tab_report:
        eda_report_ui()


if __name__ == "__main__":
    main()
