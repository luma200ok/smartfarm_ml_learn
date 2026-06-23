"""
Phase 1 · 보조 — LogisticRegression 평가 '출력'을 터미널 스타일 이미지로 저장

train_logreg.py 는 정확도/리포트를 stdout(텍스트)로만 찍어서 블로그에 넣을 그림이 없다.
이 스크립트는 같은 평가를 돌려 그 출력(정확도 + classification_report)을
맥OS 터미널 창처럼 렌더링해 figures/phase1_ml/phase1_logreg_output.png 로 저장한다.

* 영문 작물명 그대로 사용 — classification_report 의 칸 정렬(monospace)을 살리기 위함.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

np.seterr(divide="ignore", over="ignore", invalid="ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402


def build_text():
    """평가 실행 → 화면에 찍히는 것과 동일한 출력 텍스트를 만든다."""
    X_train, X_test, y_train, y_test, le, _ = prepare_data()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    lines = []
    lines.append("$ python src/ml/train_logreg.py")
    lines.append("")
    lines.append(f"  Accuracy : {acc:.4f}   ({acc:.1%})")
    lines.append("")
    lines.append("  classification report (precision / recall / f1)")
    lines.append("  " + "-" * 56)
    for ln in report.strip("\n").splitlines():
        lines.append("  " + ln)
    return lines, acc


def render(lines, acc):
    # 터미널 색상 (GitHub Dark 톤)
    BG, BAR, TEXT, GREEN, DIM = "#0d1117", "#161b22", "#e6edf3", "#3fb950", "#8b949e"

    n = len(lines)
    fig_h = 0.30 * n + 1.0
    fig = plt.figure(figsize=(9.2, fig_h), dpi=130)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # 창 배경 + 타이틀바
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                 boxstyle="round,pad=0.01,rounding_size=0.02",
                 fc=BG, ec=BAR, lw=1.5, mutation_aspect=fig_h / 9.2))
    bar_h = 0.62 / n + 0.04
    ax.add_patch(plt.Rectangle((0.02, 0.98 - bar_h), 0.96, bar_h, fc=BAR, ec="none"))
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(Circle((0.055 + i * 0.025, 0.98 - bar_h / 2), 0.008,
                     fc=c, ec="none", transform=ax.transAxes))
    ax.text(0.5, 0.98 - bar_h / 2, "LogisticRegression — 평가", color=DIM,
            ha="center", va="center", fontsize=10, family="AppleGothic")

    # 본문 텍스트 (줄 단위)
    top = 0.98 - bar_h - 0.04
    step = (top - 0.04) / max(n, 1)
    for i, ln in enumerate(lines):
        y = top - i * step
        color, weight = TEXT, "normal"
        if ln.startswith("$"):
            color = GREEN
        elif "Accuracy" in ln:
            color, weight = GREEN, "bold"
        elif ln.strip().startswith(("accuracy", "macro", "weighted")):
            color = DIM
        ax.text(0.06, y, ln, color=color, ha="left", va="top",
                fontsize=10.5, family="monospace", weight=weight)

    out = ROOT / "figures" / "phase1_ml" / "phase1_logreg_output.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"저장 완료 → {out}  (Accuracy {acc:.1%})")


if __name__ == "__main__":
    lines, acc = build_text()
    render(lines, acc)
