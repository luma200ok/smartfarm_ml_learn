"""
Phase 1 · 청크 1-2 — 데이터 로드 + 첫 탐색 (EDA 1단계)

목표: 데이터가 '어떻게 생겼는지' 감을 잡는다.
  - 모양(행·열), 미리보기, 자료형, 기초 통계, 결측치, 클래스(작물) 분포
"""
from pathlib import Path
import pandas as pd

# 출력 잘리지 않게 (학습용 — 컬럼 다 보이게)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

# 1) 경로: 이 파일(src/ml/) 기준으로 프로젝트 루트를 찾아 data/ 로 (철자 정확히!)
ROOT = Path(__file__).resolve().parents[2]          # smartfarm_ai/
CSV = ROOT / "data" / "Crop_recommendation.csv"

# 2) CSV 읽기 → DataFrame(표)
df = pd.read_csv(CSV)

print("=" * 60)
print("① 모양 (행, 열):", df.shape)                  # (샘플 수, 컬럼 수)

print("\n" + "=" * 60)
print("② 앞부분 미리보기 (head)")
print(df.head())

print("\n" + "=" * 60)
print("③ 컬럼 정보 (이름 · 자료형 · 결측 여부)")
df.info()

print("\n" + "=" * 60)
print("④ 기초 통계 (숫자 컬럼: 평균·표준편차·min·max 등)")
print(df.describe())

print("\n" + "=" * 60)
print("⑤ 결측치 개수 (컬럼별 비어있는 칸)")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("⑥ 정답(label) = 작물 종류")
print("종류 수:", df["label"].nunique())
print(df["label"].value_counts())
