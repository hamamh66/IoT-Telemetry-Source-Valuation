"""Runs all experiments on RT-IoT2022 and writes raw and summary results.
Usage: python run_rtiot.py path/to/rt-iot2022.csv [outputs_root]"""
import sys, time, pickle, numpy as np, pandas as pd
from pathlib import Path
import rtiot_xai as r
DATA = sys.argv[1] if len(sys.argv) > 1 else "rt-iot2022.csv"
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "Outputs"
TAB, OTH = ROOT / "Tables", ROOT / "Others"; TAB.mkdir(parents=True, exist_ok=True); OTH.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def stamp(m): print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
AU, CO, PC, SEL, CS, DG, DGP, CMs = [], [], [], [], [], [], [], []
for seed in r.SEEDS:
    stamp(f"seed {seed}: start")
    AU.append(r.audit(DATA, seed))
    c, pc, sel, cm, models, data = r.compact(DATA, seed)
    CO.append(c); PC.append(pc); SEL.append(sel); CMs.append(cm)
    d, dp = r.degraded(models, data, seed); DG.append(d); DGP.append(dp)
    CS.append(r.coalitions(DATA, seed))
    if seed == r.SEEDS[0]:   # keep the first split's compact model for the explanation figures
        m_, cols = models["two-pass, SHAP top-20"]
        pickle.dump(dict(model=m_, cols=cols, Xtr=data[0][cols], Xte=data[1][cols], ytr=data[2], yte=data[3],
                         classes=data[4]), open(OTH / "seed0_compact_model.pkl", "wb"))
    stamp(f"seed {seed}: done")
au = pd.concat(AU); co = pd.concat(CO); pc = pd.concat(PC); sel = pd.concat(SEL); cs = pd.concat(CS)
dg = pd.concat(DG); dgp = pd.concat(DGP)
for name, df in [("audit_raw", au), ("compact_raw", co), ("per_class_raw", pc), ("selected_features_raw", sel),
                 ("coalitions_raw", cs), ("degraded_raw", dg), ("degraded_per_class_raw", dgp)]:
    df.to_csv(OTH / f"{name}.csv", index=False)
np.save(OTH / "confusion_matrices_two_pass.npy", np.stack(CMs))
# ---- summaries with 95 % confidence intervals over the repeated splits
def summ(df, keys, cols):
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        for c in cols:
            m, lo, hi = r.ci(g[c]); row[c] = m; row[c + "_lo"] = lo; row[c + "_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
M = ["macro_f1", "accuracy", "attack_recall", "attack_precision"]
summ(au, ["index_column", "model"], M).to_csv(TAB / "audit_summary.csv", index=False)
au.groupby(["index_column", "model"]).index_in_top20.mean().reset_index().to_csv(TAB / "audit_index_in_top20.csv", index=False)
summ(co, ["model"], M + ["n_features", "n_groups", "fit_s", "infer_us_per_flow"]).to_csv(TAB / "compact_summary.csv", index=False)
summ(pc, ["model", "class"], ["precision", "recall", "f1", "support"]).to_csv(TAB / "per_class_summary.csv", index=False)
sel.groupby(["feature", "group"]).agg(times_selected=("seed", "count"), mean_rank=("rank", "mean")).reset_index() \
   .sort_values(["times_selected", "mean_rank"], ascending=[False, True]).to_csv(TAB / "selected_features_stability.csv", index=False)
summ(cs, ["coalition"], M).to_csv(TAB / "coalitions_summary.csv", index=False)
pd.concat([r.shapley(cs, m) for m in M]).to_csv(TAB / "stream_shapley.csv", index=False)
summ(dg, ["model", "missing_source"], M).to_csv(TAB / "degraded_summary.csv", index=False)
summ(dgp, ["model", "missing_source", "class"], ["recall", "f1"]).to_csv(TAB / "degraded_per_class_summary.csv", index=False)
stamp("all done")
