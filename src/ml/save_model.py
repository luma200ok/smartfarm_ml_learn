"""
Phase 1 · 청크 1-8a — 베스트 모델(RandomForest) 저장

학습 ↔ 서빙 분리: 여기서 딱 1번 학습 → .pkl 파일로 저장.
Streamlit 데모(1-8b)는 이 파일을 '불러와' 예측만 한다 (다시 학습 X).

★ 저장 묶음 3종 (예측에 다 필요) ─────────────────────
  - model  : RandomForest   (환경값 → 작물번호)
  - scaler : StandardScaler (사용자 입력을 '학습 때와 같은 잣대'로 변환)
  - le     : LabelEncoder   (작물번호 0~21 → 작물이름 'rice' 되돌리기)
모델만 저장하면 입력 변환·이름 복원을 못 해서 예측이 안 됨.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
CSV = ROOT / "data" / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    # 1. 데이터 준비 (1-4 재사용) — scaler·le 도 같이 받음
    X_train, X_test, y_train, y_test, le, scaler = prepare_data()

    # 2. 베스트 모델(1-7 비교 결과 = RandomForest) 학습 — 딱 1번
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. 작물별 환경 프로파일 (평균·최소·최대) — 데모 '환경 가이드'용
    #    원본 CSV는 배포에 안 올리므로, 집계값만 묶음에 저장해서 같이 보냄
    df = pd.read_csv(CSV)
    prof_mean = df.groupby("label")[FEATURES].mean().round(1)
    prof_min = df.groupby("label")[FEATURES].min().round(1)
    prof_max = df.groupby("label")[FEATURES].max().round(1)

    # 4. 묶어서 저장 — joblib.dump(객체, 경로)
    bundle = {
        "model": model, "scaler": scaler, "le": le,
        "prof_mean": prof_mean, "prof_min": prof_min, "prof_max": prof_max,
    }
    out = MODELS / "phase1_crop_rf.pkl"
    joblib.dump(bundle, out)

    print("저장 완료 →", out)
    print(f"  - model : {type(model).__name__} (트리 {model.n_estimators}그루)")
    print(f"  - scaler: {type(scaler).__name__}")
    print(f"  - le    : 작물 {len(le.classes_)}종")
    print(f"  - 프로파일: 작물 {len(prof_mean)}종 × 피처 {len(FEATURES)}개 (평균·최소·최대)")


if __name__ == "__main__":
    main()
