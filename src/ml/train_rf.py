"""
Phase 1 · 청크 1-6 — 모델 학습 ② RandomForest (LogReg와 비교)

LogReg(1-5, '직선'으로만 가름)가 못 떼낸 rice↔jute 헷갈림을,
RandomForest('결정나무 여러 그루의 투표')가 개선하는지 확인한다.

1-5와 5단계 골격 동일 — 모델 한 줄만 교체.
+ feature importance(어떤 피처가 작물 가르기에 중요한가) 추가.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402  (1-4 전처리 재사용)
from common.labels_ko import ko_crop, FEATURE_KO   # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 1-5 LogReg 기준선 (비교용)
LOGREG_ACC = 0.9727

# 피처 순서 = CSV 컬럼 순서(label 제외). prepare_data가 이 순서로 X를 만듦.
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    # 1. 데이터 준비 (청크 1-4 재사용) -----------------------------
    X_train, X_test, y_train, y_test, le, scaler = prepare_data()

    # 2. 모델 학습 — RandomForest
    #    n_estimators=200 : 결정나무 200그루 (많을수록 안정, 느려짐)
    #    n_jobs=-1        : CPU 전부 써서 병렬 학습
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. 예측
    y_pred = model.predict(X_test)

    # 4. 평가 ① 정확도 + LogReg와 비교 ----------------------------
    acc = accuracy_score(y_test, y_pred)
    print("=" * 55)
    print(f"① 정확도(Accuracy): {acc:.4f}  ({acc:.1%})")
    print(f"   ↳ 1-5 LogReg : {LOGREG_ACC:.4f}  → 차이 {acc - LOGREG_ACC:+.4f}")

    # 5. 평가 ② 작물별 상세
    print("\n" + "=" * 55)
    print("② 작물별 상세 평가 (precision/recall/f1)")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 6. 평가 ③ 혼동행렬 그림 -------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    labels_ko = [ko_crop(c) for c in le.classes_]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels_ko, yticklabels=labels_ko,
                cbar=False, square=True)
    plt.xlabel("예측")
    plt.ylabel("실제")
    plt.title("③ 혼동행렬 — RandomForest", fontsize=14)
    plt.tight_layout()
    FIG = ROOT / "figures"
    FIG.mkdir(exist_ok=True)
    plt.savefig(FIG / "phase1_rf_confusion.png", dpi=100)
    plt.close()

    # 7. feature importance — 어떤 피처가 작물 가르기에 중요한가 ----
    #    (RandomForest만의 보너스. LogReg엔 이 형태가 없음)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]   # 큰 순서로 정렬
    print("=" * 55)
    print("④ 피처 중요도 (작물 가르기 기여도)")
    for i in order:
        print(f"   {FEATURE_KO[FEATURES[i]]:10s} {importances[i]:.3f}")

    names_ko = [FEATURE_KO[FEATURES[i]] for i in order]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[order], y=names_ko, color="#4C9A2A")
    plt.xlabel("중요도")
    plt.title("④ 피처 중요도 — RandomForest", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "phase1_rf_importance.png", dpi=100)
    plt.close()
    print("=" * 55)
    print(f"그림 2장 → {FIG}/phase1_rf_confusion.png, phase1_rf_importance.png")


if __name__ == "__main__":
    main()
