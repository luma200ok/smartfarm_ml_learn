"""
Phase 1 · 청크 1-7 — 모델 학습 ③ XGBoost + 세 모델 비교 (베스트 선정)

A. XGBoost 학습·평가 (1-5/1-6과 같은 5단계 골격 + 피처 중요도)
B. LogReg / RandomForest / XGBoost 정확도 비교표 → 데모에 쓸 베스트 모델 확정

XGBoost = '부스팅' 트리. RandomForest(나무들 독립 투표)와 달리,
앞 나무의 실수를 다음 나무가 보완하며 순차로 키운다(오답에 집중).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402
from common.labels_ko import ko_crop, FEATURE_KO   # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# macOS numpy(Accelerate BLAS) 헛 경고 무시 (LogReg 학습 시 뜸, 결과엔 무관)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FIG = ROOT / "figures"


def main():
    X_train, X_test, y_train, y_test, le, scaler = prepare_data()

    # === A. XGBoost 학습·평가 ====================================
    xgb = XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred)

    print("=" * 55)
    print(f"① XGBoost 정확도: {acc_xgb:.4f}  ({acc_xgb:.1%})")
    print("\n② 작물별 상세 평가")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 혼동행렬 그림
    cm = confusion_matrix(y_test, y_pred)
    labels_ko = [ko_crop(c) for c in le.classes_]
    FIG.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels_ko, yticklabels=labels_ko,
                cbar=False, square=True)
    plt.xlabel("예측"); plt.ylabel("실제")
    plt.title("③ 혼동행렬 — XGBoost", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "phase1_xgb_confusion.png", dpi=100)
    plt.close()

    # 피처 중요도
    imp = xgb.feature_importances_
    order = np.argsort(imp)[::-1]
    print("=" * 55)
    print("④ 피처 중요도 (XGBoost)")
    for i in order:
        print(f"   {FEATURE_KO[FEATURES[i]]:10s} {imp[i]:.3f}")

    names_ko = [FEATURE_KO[FEATURES[i]] for i in order]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=imp[order], y=names_ko, color="#4C9A2A")
    plt.xlabel("중요도")
    plt.title("④ 피처 중요도 — XGBoost", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "phase1_xgb_importance.png", dpi=100)
    plt.close()

    # === B. 세 모델 비교 (정직하게 다 학습해서 뽑기) =============
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":            XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        scores[name] = accuracy_score(y_test, m.predict(X_test))

    best = max(scores, key=scores.get)
    print("\n" + "=" * 55)
    print("⑤ 세 모델 비교 (베스트 선정)")
    for name, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        mark = " ⭐ 베스트" if name == best else ""
        print(f"   {name:20s} {s:.4f}  ({s:.1%}){mark}")

    # 비교 막대그래프
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    plt.figure(figsize=(7, 4))
    bars = sns.barplot(x=names, y=vals, color="#A0D995")
    plt.ylim(0.9, 1.0)   # 차이가 잘 보이게 90%부터
    for i, v in enumerate(vals):
        bars.text(i, v + 0.002, f"{v:.3f}", ha="center")
    plt.ylabel("정확도")
    plt.title("⑤ 모델별 정확도 비교", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "phase1_model_compare.png", dpi=100)
    plt.close()

    print("=" * 55)
    print(f"베스트 모델 = {best} ({scores[best]:.1%}) → 1-8 Streamlit 데모에 사용")
    print(f"그림 → {FIG} : phase1_xgb_confusion.png, phase1_xgb_importance.png, phase1_model_compare.png")


if __name__ == "__main__":
    main()
