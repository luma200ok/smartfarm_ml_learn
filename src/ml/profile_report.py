"""
Phase 1 · YData Profiling 자동 EDA 리포트

Crop_recommendation.csv 를 ydata-profiling 으로 한 번에 프로파일링한다.
  - HTML 리포트 저장 (reports/phase1_eda_profile.html) — 인터랙티브 확인용
  - 핵심 통계 JSON 저장 (reports/phase1_eda_profile.json) — 수행내역서 md 작성용

실행: python src/ml/profile_report.py
"""
from pathlib import Path
import json
import warnings

import pandas as pd
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "Crop_recommendation.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(CSV)

    # 분류 문제: label(작물)을 타깃으로 지정하면 타깃 연관 분석이 추가됨
    profile = ProfileReport(
        df,
        title="스마트팜 작물 추천 — Phase 1 EDA (YData Profiling)",
        explorative=True,
    )

    html_out = REPORTS / "phase1_eda_profile.html"
    profile.to_file(html_out)
    print("HTML 리포트 저장 →", html_out)

    # 핵심 통계 추출 (md 작성용)
    desc = profile.get_description()
    table = desc.table  # 전체 표 요약 (행/열/결측/중복 등)
    variables = desc.variables  # 변수별 통계

    summary = {
        "n_obs": int(table["n"]),
        "n_var": int(table["n_var"]),
        "n_cells_missing": int(table["n_cells_missing"]),
        "p_cells_missing": float(table["p_cells_missing"]),
        "n_duplicates": int(table.get("n_duplicates", 0)),
        "types": {str(k): int(v) for k, v in table.get("types", {}).items()},
        "variables": {},
        "alerts": [str(a) for a in desc.alerts],
    }

    for name, v in variables.items():
        item = {"type": str(v.get("type"))}
        for key in ["n_missing", "n_distinct", "mean", "std", "min", "max",
                    "5%", "25%", "50%", "75%", "95%"]:
            if key in v and v[key] is not None:
                try:
                    item[key] = round(float(v[key]), 2)
                except (TypeError, ValueError):
                    item[key] = v[key]
        summary["variables"][str(name)] = item

    json_out = REPORTS / "phase1_eda_profile.json"
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("핵심 통계 JSON 저장 →", json_out)
    print(f"  행 {summary['n_obs']} · 열 {summary['n_var']} · 결측 {summary['n_cells_missing']} · 중복 {summary['n_duplicates']}")
    print(f"  경고(alerts) {len(summary['alerts'])}건")


if __name__ == "__main__":
    main()
