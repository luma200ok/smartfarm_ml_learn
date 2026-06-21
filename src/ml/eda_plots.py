"""
Phase 1 · EDA 시각화 (그래프 = 한글 통일)

그림 3장 (모두 한글 라벨):
  ① 피처 분포(histogram)
  ② 상관관계 heatmap
  ③ 작물별 강수량 boxplot
+ 작물 영↔한 대조표, 핵심 수치 출력 (터미널은 영어 원본 유지)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.labels_ko import FEATURE_KO, ko_crop  # noqa: E402

# 한글 깨짐 방지 (그래프 전용)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT / "data" / "Crop_recommendation.csv")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ① 피처 분포 (컬럼명 한글로 바꿔서 표시) -----------------------
df[features].rename(columns=FEATURE_KO).hist(bins=30, figsize=(12, 8), color="#4C9A2A")
plt.suptitle("① 피처 분포", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "phase1_distributions.png", dpi=100)
plt.close()

# ② 상관관계 heatmap (축 라벨 한글) ----------------------------
corr = df[features].corr()
corr_ko = corr.rename(index=FEATURE_KO, columns=FEATURE_KO)
plt.figure(figsize=(8, 7))
sns.heatmap(corr_ko, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("② 피처 상관관계", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "phase1_corr.png", dpi=100)
plt.close()

# ③ 작물별 강수량 (x축 작물명 한글, 중앙값 순) ------------------
order = df.groupby("label")["rainfall"].median().sort_values().index
plt.figure(figsize=(13, 6))
sns.boxplot(data=df, x="label", y="rainfall", order=order, color="#A0D995")
plt.xticks(range(len(order)), [ko_crop(c) for c in order], rotation=90)
plt.xlabel("작물")
plt.ylabel("강수량(mm)")
plt.title("③ 작물별 강수량 분포", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "phase1_rainfall_by_crop.png", dpi=100)
plt.close()

# --- 터미널 출력 (영어 원본 + 대조표) -------------------------
print("=== 상관계수 0.3 이상 (눈에 띄는 연관) ===")
pairs = corr.abs().unstack().sort_values(ascending=False)
pairs = pairs[pairs < 1].drop_duplicates()
print(pairs[pairs >= 0.3].round(2).to_string() or "(없음)")

print("\n=== 작물 영↔한 대조표 (강수량 평균 낮은 순) ===")
rain_mean = df.groupby("label")["rainfall"].mean().sort_values()
for en, rain in rain_mean.items():
    print(f"  {en:12s} → {ko_crop(en):8s} (강수 {rain:5.0f}mm)")

print("\n저장된 그림 3장 (한글) →", FIG)
