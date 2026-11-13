"""Supervisory review routing: route the least confident alerts of the two-pass forest to an analyst and
measure how many classification errors the routed share contains (5 stratified splits).
Uses the SHAP-selected feature sets saved by run_rtiot.py, so the forests are identical to those reported.
Usage: python review_routing.py path/to/rt-iot2022.csv [outputs_root]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
import rtiot_xai as r
DATA = sys.argv[1] if len(sys.argv) > 1 else "rt-iot2022.csv"
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "Outputs"
sel = pd.read_csv(ROOT / "Others" / "selected_features_raw.csv")
X, y = r.load(DATA); rows = []
for seed in r.SEEDS:
    cols = sel[sel.seed == seed].sort_values("rank").feature.tolist()
    Xtr, Xte, ytr, yte = r.split(X, y, seed)
    m = r.rf(seed).fit(Xtr[cols], ytr)
    proba = m.predict_proba(Xte[cols]); pred = m.classes_[proba.argmax(1)]; conf = proba.max(1)
    err = pred != yte.values; order = np.argsort(conf)            # least confident first
    for cov in (0.0, 0.005, 0.01, 0.02, 0.05):
        k = int(round(cov * len(conf))); routed = np.zeros(len(conf), bool); routed[order[:k]] = True
        # an analyst who reviews a routed alert is assumed to correct it (upper bound on the benefit of review)
        f1_after = r.f1_score(yte.values, np.where(routed, yte.values, pred), average="macro", zero_division=0)
        rows.append(dict(seed=seed, review_share=cov, routed_flows=k, errors_total=int(err.sum()),
                         errors_routed=int(err[routed].sum()), share_of_errors_routed=err[routed].sum() / max(err.sum(), 1),
                         macro_f1_after_review=f1_after, confident_errors_p_ge_0_99=int((err & (conf >= 0.99)).sum())))
    print(f"  routing seed={seed} done", flush=True)
raw = pd.DataFrame(rows); raw.to_csv(ROOT / "Others" / "review_routing_raw.csv", index=False)
out = []
for cov, g in raw.groupby("review_share"):
    row = dict(review_share=cov, routed_flows=g.routed_flows.mean(), errors_total=g.errors_total.mean())
    for c in ("share_of_errors_routed", "macro_f1_after_review", "confident_errors_p_ge_0_99"):
        m_, lo, hi = r.ci(g[c]); row[c] = m_; row[c + "_lo"] = lo; row[c + "_hi"] = hi
    out.append(row)
pd.DataFrame(out).to_csv(ROOT / "Tables" / "review_routing_summary.csv", index=False)
print(pd.DataFrame(out).round(4).to_string())
