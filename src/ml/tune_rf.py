"""
Phase 1 · 보강 — RandomForest 하이퍼파라미터 튜닝 + 과적합(Train vs Test) 점검

수행내역서 '4.4 하이퍼파라미터 튜닝' / '과적합 점검' 섹션에 넣을 실수치를 만든다.
  ① 베이스라인(기본값) Train/Test 정확도 → 과적합 점검
  ② GridSearchCV(5-fold)로 최적 조합 탐색 → 튜닝 전/후 비교
값은 stdout으로 출력 (표로 보고서에 옮김).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

np.seterr(divide="ignore", over="ignore", invalid="ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml.preprocess import prepare_data   # noqa: E402


def evaluate(model, X_tr, X_te, y_tr, y_te):
    tr = accuracy_score(y_tr, model.predict(X_tr))
    te = accuracy_score(y_te, model.predict(X_te))
    f1 = f1_score(y_te, model.predict(X_te), average="macro")
    return tr, te, f1


def main():
    X_train, X_test, y_train, y_test, le, _ = prepare_data()

    # ① 베이스라인 (현재 코드 기본값) -----------------------------
    base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    base.fit(X_train, y_train)
    b_tr, b_te, b_f1 = evaluate(base, X_train, X_test, y_train, y_test)

    print("=" * 60)
    print("① 베이스라인 RandomForest (n_estimators=200, 기본값)")
    print(f"   Train Acc : {b_tr:.4f}")
    print(f"   Test  Acc : {b_te:.4f}   (Gap {b_tr - b_te:+.4f})")
    print(f"   Test  F1  : {b_f1:.4f}")

    # ② GridSearchCV 튜닝 -----------------------------------------
    grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        grid, cv=5, scoring="accuracy", n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    t_tr, t_te, t_f1 = evaluate(best, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("② GridSearchCV 튜닝 결과 (5-fold, 후보 36조합)")
    print(f"   best params : {gs.best_params_}")
    print(f"   best CV Acc : {gs.best_score_:.4f}")
    print(f"   Train Acc   : {t_tr:.4f}")
    print(f"   Test  Acc   : {t_te:.4f}   (Gap {t_tr - t_te:+.4f})")
    print(f"   Test  F1    : {t_f1:.4f}")

    # ③ 튜닝 전/후 요약 표 ----------------------------------------
    print("\n" + "=" * 60)
    print("③ 튜닝 전/후 비교")
    print(f"{'지표':<12}{'튜닝 전':>12}{'튜닝 후':>12}{'변화':>12}")
    for name, b, t in [("Test Acc", b_te, t_te),
                       ("Test F1", b_f1, t_f1),
                       ("Train-Test Gap", b_tr - b_te, t_tr - t_te)]:
        print(f"{name:<12}{b:>12.4f}{t:>12.4f}{t - b:>+12.4f}")


if __name__ == "__main__":
    main()
