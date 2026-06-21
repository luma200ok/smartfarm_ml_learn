"""
Phase 1 · 청크 1-5 — 모델 학습 ① LogisticRegression + 평가

흐름: prepare_data()(1-4 재사용) → 학습 → 예측 → 평가 3종
  ① Accuracy              : 전체 정답률 (한 숫자)
  ② classification_report : 작물별 정밀도/재현율/F1 (어떤 작물이 약한지)
  ③ confusion matrix      : 무엇을 무엇으로 헷갈렸나 (그림 저장)

모델: LogisticRegression — 분류 입문용. '선'으로 클래스를 가르는 가장 기본 모델.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# macOS numpy(Accelerate BLAS)가 행렬곱 때 띄우는 '헛 경고'(divide by zero/overflow)
# 무시 — 계산 결과는 정상, 화면만 어지럽혀서 끔. (코드 버그 아님)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402  (1-4 전처리 재사용)
from common.labels_ko import ko_crop      # noqa: E402

# 한글 깨짐 방지 (그래프 전용)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 1. 데이터 준비 (청크 1-4 그대로 재사용) -----------------------
    X_train, X_test, y_train, y_test, le, scaler = prepare_data()

    # 2. 모델 학습 — 문제(X_train) + 정답(y_train) 보여주며 규칙 익히기
    #    max_iter=1000 : 최적값 찾는 반복 횟수 (기본 100이면 수렴 경고 날 수 있어 늘림)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 3. 예측 — 시험지(X_test)만 주고 정답 맞히게 (y_test는 안 보여줌)
    y_pred = model.predict(X_test)

    # 4. 평가 ① 정확도 — 440개 중 몇 개 맞혔나 ----------------------
    acc = accuracy_score(y_test, y_pred)
    print("=" * 55)
    print(f"① 정확도(Accuracy): {acc:.4f}  ({acc:.1%})")

    # 5. 평가 ② 작물별 상세 — 정밀도/재현율/F1 (어떤 작물이 약점인지)
    print("\n" + "=" * 55)
    print("② 작물별 상세 평가 (precision/recall/f1)")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 6. 평가 ③ 혼동행렬 그림 — 실제(세로) vs 예측(가로) ------------
    #    대각선=정답, 대각선 밖 숫자=헷갈린 것
    cm = confusion_matrix(y_test, y_pred)
    labels_ko = [ko_crop(c) for c in le.classes_]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels_ko, yticklabels=labels_ko,
                cbar=False, square=True)
    plt.xlabel("예측")
    plt.ylabel("실제")
    plt.title("③ 혼동행렬 — LogisticRegression", fontsize=14)
    plt.tight_layout()

    FIG = ROOT / "figures"
    FIG.mkdir(exist_ok=True)
    out = FIG / "phase1_logreg_confusion.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print("=" * 55)
    print(f"③ 혼동행렬 그림 →{out}")


if __name__ == "__main__":
    main()
