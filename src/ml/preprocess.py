"""
Phase 1 · 청크 1-4 — 전처리 (모델 학습 직전 준비)

3가지 처리:
  ① 인코딩      : label 문자(rice...) → 숫자 (모델은 숫자만 먹음)
  ② train/test  : 학습용/평가용 분리 (stratify=클래스 비율 유지)
  ③ 스케일링    : 피처 범위 통일 — train에만 fit (★데이터 누수 방지)

prepare_data() 함수로 만들어 1-5(모델 학습)에서 재사용한다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "Crop_recommendation.csv"


def prepare_data(test_size=0.2, random_state=42, scale=True):
    df = pd.read_csv(CSV)

    X = df.drop(columns=["label"])   # 입력 피처 7개
    y_raw = df["label"]              # 정답 (문자)

    # ① 인코딩: 문자 작물 → 숫자 (rice, maize ... → 0~21)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ② train/test 분리 — stratify=y 로 22종 비율을 양쪽에 똑같이 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ③ 스케일링 — train에만 fit, test는 그 기준으로 transform만
    #    (test로 fit하면 평가 데이터 정보가 새어들어감 = 데이터 누수)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, scaler = prepare_data()

    print("=" * 55)
    print("① 인코딩 — 문자 → 숫자 (앞 5개)")
    for name, code in list(zip(le.classes_, range(len(le.classes_))))[:5]:
        print(f"   {name:12s} → {code}")
    print(f"   ... 총 {len(le.classes_)}종")

    print("\n" + "=" * 55)
    print("② train/test 분리")
    print(f"   train: {X_train.shape[0]}개  /  test: {X_test.shape[0]}개")
    print(f"   (비율 {1 - 0.2:.0%} : {0.2:.0%}, 피처 {X_train.shape[1]}개)")

    print("\n" + "=" * 55)
    print("③ 스케일링 효과 — train 각 피처 평균≈0, 표준편차≈1")
    print("   평균:", np.round(X_train.mean(axis=0), 2))
    print("   표준편차:", np.round(X_train.std(axis=0), 2))
    print("   → 범위 제각각이던 7개 피처가 같은 잣대로 맞춰짐")
