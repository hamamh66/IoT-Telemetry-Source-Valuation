"""Figures and LaTeX-ready table values for the RT-IoT2022 study.
Usage: python make_figures.py [outputs_root]"""
import sys, pickle, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import rtiot_xai as r
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "Outputs"
TAB, OTH, FIG = ROOT / "Tables", ROOT / "Others", ROOT / "Figures"; FIG.mkdir(parents=True, exist_ok=True)
GLAB = {"timing": "Timing", "volume": "Volume and payload", "header": "Header, flags, window",
        "protocol": "Protocol and service", "endpoint": "Endpoint ports"}
GCOL = {"timing": "#2E5F8A", "volume": "#B7770D", "header": "#1E8449", "protocol": "#7D3C98", "endpoint": "#C0392B"}
def tidy(ax):
    ax.grid(color="#e6e6e6", axis="x"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---- Fig. 1: leakage audit
a = pd.read_csv(TAB / "audit_summary.csv"); c = pd.read_csv(TAB / "compact_summary.csv")
rows = [("Regressor + rounding, index kept\n(original pipeline)", a.query("index_column and model.str.startswith('regressor')", engine="python")),
        ("Regressor + rounding, index removed", a.query("not index_column and model.str.startswith('regressor')", engine="python")),
        ("Classifier, index kept", a.query("index_column and model.str.startswith('classifier')", engine="python")),
        ("Classifier, index removed,\nSHAP on unbalanced sample", c[c.model == "two-pass, SHAP top-20 (unbalanced sample)"]),
        ("Classifier, index removed,\nSHAP on class-balanced sample", c[c.model == "two-pass, SHAP top-20"]),
        ("Classifier, all 83 features", c[c.model == "full (83 features)"])]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
for i, (lab, d) in enumerate(rows):
    m, lo, hi = d.macro_f1.iloc[0], d.macro_f1_lo.iloc[0], d.macro_f1_hi.iloc[0]
    col = "#9AA0A6" if i < 3 else "#2E5F8A"
    ax.barh(i, m, color=col, height=0.6); ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color="k", capsize=3, lw=1)
    ax.text(min(hi, 1.0) + 0.004, i, f"{m:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([x[0] for x in rows], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0.75, 1.03); ax.set_xlabel("Macro F1 (mean and 95% CI over 5 stratified splits)"); tidy(ax)
fig.tight_layout(); fig.savefig(FIG / "fig1_leakage_audit.png", dpi=300); plt.close(fig)

# ---- Fig. 2: global SHAP ranking of the compact model (seed 0) by telemetry source
sel = pd.read_csv(OTH / "selected_features_raw.csv"); s0 = sel[sel.seed == 0].sort_values("rank")
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.barh(range(len(s0)), s0.mean_abs_shap, color=[GCOL[g] for g in s0.group])
ax.set_yticks(range(len(s0))); ax.set_yticklabels(s0.feature, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP| over a class-balanced training sample (first-pass forest)"); tidy(ax)
for g in r.GROUPS: ax.barh([], [], color=GCOL[g], label=GLAB[g])
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig2_shap_selection.png", dpi=300); plt.close(fig)

# ---- Fig. 3: per-class F1 with CI, full vs compact
p = pd.read_csv(TAB / "per_class_summary.csv")
cls = p[p.model == "full (83 features)"].sort_values("support", ascending=False)["class"].tolist()
fig, ax = plt.subplots(figsize=(7.4, 4.6)); w = 0.38
for j, (mname, col, lab) in enumerate([("full (83 features)", "#9AA0A6", "All 83 features"),
                                       ("two-pass, SHAP top-20", "#2E5F8A", "Two-pass, 20 features")]):
    d = p[p.model == mname].set_index("class").loc[cls]
    y = np.arange(len(cls)) + (j - 0.5) * w
    ax.barh(y, d.f1, height=w, color=col, label=lab)
    ax.errorbar(d.f1, y, xerr=[d.f1 - d.f1_lo, (d.f1_hi - d.f1).clip(lower=0)], fmt="none", color="k", capsize=2, lw=0.8)
lab = [f"{k}{' (normal)' if k in r.NORMAL else ''}  n={int(s)}" for k, s in
       zip(cls, p[p.model == "full (83 features)"].set_index("class").loc[cls].support)]
ax.set_yticks(range(len(cls))); ax.set_yticklabels(lab, fontsize=7.5); ax.invert_yaxis()
ax.set_xlim(0.6, 1.02); ax.set_xlabel("Per-class F1 (mean and 95% CI over 5 splits; n = test support)"); tidy(ax)
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2)
fig.tight_layout(); fig.savefig(FIG / "fig3_per_class_f1.png", dpi=300); plt.close(fig)

# ---- Fig. 4: Shapley value of each telemetry source
sh = pd.read_csv(TAB / "stream_shapley.csv"); sh = sh[sh.metric == "macro_f1"].set_index("group").loc[r.GROUPS]
fig, ax = plt.subplots(figsize=(7.2, 3.8)); y = np.arange(len(sh)); w = 0.27
ax.barh(y - w, sh.alone, height=w, color="#D5D8DC", label="Contribution alone")
ax.barh(y, sh.shapley, height=w, color=[GCOL[g] for g in sh.index], label="Shapley value")
ax.errorbar(sh.shapley, y, xerr=[sh.shapley - sh.ci_low, sh.ci_high - sh.shapley], fmt="none", color="k", capsize=2, lw=0.8)
ax.barh(y + w, sh.added_last, height=w, color="#4A4F54", label="Contribution when added last")
ax.set_yticks(y); ax.set_yticklabels([GLAB[g] for g in sh.index], fontsize=8.5); ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("Macro-F1 points attributed to the source"); tidy(ax)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig4_source_shapley.png", dpi=300); plt.close(fig)

# ---- Fig. 5: runtime loss of a source vs a fallback model trained without it
dg = pd.read_csv(TAB / "degraded_summary.csv"); co = pd.read_csv(TAB / "coalitions_summary.csv").set_index("coalition")
key = lambda S: "+".join(g for g in r.GROUPS if g in S)
fig, ax = plt.subplots(figsize=(7.4, 4.2)); y = np.arange(len(r.GROUPS)); w = 0.27
for j, (mname, col, lab) in enumerate([("full (83 features)", "#9AA0A6", "All-feature forest, source imputed"),
                                       ("two-pass, SHAP top-20", "#2E5F8A", "Two-pass forest, source imputed")]):
    d = dg[dg.model == mname].set_index("missing_source").loc[r.GROUPS]
    ax.barh(y + (j - 1) * w, d.macro_f1, height=w, color=col, label=lab)
    ax.errorbar(d.macro_f1, y + (j - 1) * w, xerr=[d.macro_f1 - d.macro_f1_lo, d.macro_f1_hi - d.macro_f1], fmt="none", color="k", capsize=2, lw=0.7)
fb = co.loc[[key([o for o in r.GROUPS if o != g]) for g in r.GROUPS]]
ax.barh(y + w, fb.macro_f1.values, height=w, color="#1E8449", label="Fallback forest trained without the source")
ax.errorbar(fb.macro_f1.values, y + w, xerr=[fb.macro_f1.values - fb.macro_f1_lo.values, fb.macro_f1_hi.values - fb.macro_f1.values], fmt="none", color="k", capsize=2, lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([f"{GLAB[g]} unavailable" for g in r.GROUPS], fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(0, 1.05); ax.set_xlabel("Macro F1 (mean and 95% CI over 5 splits)"); tidy(ax)
ax.legend(fontsize=7.2, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
fig.tight_layout(); fig.savefig(FIG / "fig5_degraded_vs_fallback.png", dpi=300); plt.close(fig)

# ---- Fig. 6: confusion matrix of the two-pass model, summed over the 5 test splits
cms = np.load(OTH / "confusion_matrices_two_pass.npy").sum(axis=0)
labels = sorted(p["class"].unique())
fig, ax = plt.subplots(figsize=(7.2, 6.0))
norm = cms / cms.sum(axis=1, keepdims=True)
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cms[i, j]: ax.text(j, i, f"{cms[i, j]}", ha="center", va="center", fontsize=6.5, color="white" if norm[i, j] > 0.5 else "black")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); fig.colorbar(im, fraction=0.04, label="Row-normalized rate")
fig.tight_layout(); fig.savefig(FIG / "fig6_confusion_two_pass.png", dpi=300); plt.close(fig)

# ---- Fig. 7 and the alert record: local SHAP for one correct and one misclassified test flow (seed 0)
import shap
S = pickle.load(open(OTH / "seed0_compact_model.pkl", "rb"))
m_, cols, Xte, yte, classes = S["model"], S["cols"], S["Xte"], S["yte"], S["classes"]
pred = m_.predict(Xte); proba = m_.predict_proba(Xte); ex = shap.TreeExplainer(m_)
wrong = np.flatnonzero((pred != yte.values) & ~yte.isin(["DOS_SYN_Hping"]).values)
pick_wrong = wrong[np.argmax(proba[wrong].max(axis=1))] if len(wrong) else None
right = np.flatnonzero((pred == yte.values) & (yte.values == "NMAP_TCP_scan"))
pick_right = right[0]
records = []
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
for ax, i, title in [(axes[0], pick_right, "Correct alert"), (axes[1], pick_wrong, "Misclassified flow")]:
    x = Xte.iloc[[i]]; sv = np.asarray(ex.shap_values(x, check_additivity=False))
    k = list(m_.classes_).index(pred[i])
    contrib = sv[0, :, k] if sv.ndim == 3 else sv[k][0]
    o = np.argsort(-np.abs(contrib))[:8]
    ax.barh(range(len(o)), contrib[o], color=["#C0392B" if v > 0 else "#2E5F8A" for v in contrib[o]])
    ax.set_yticks(range(len(o))); ax.set_yticklabels([f"{cols[j]} = {x.iloc[0, j]:.4g}" for j in o], fontsize=7.5); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); tidy(ax)
    ax.set_title(f"{title}: true {yte.iloc[i]}, predicted {pred[i]} (p = {proba[i].max():.2f})", fontsize=8.5)
    ax.set_xlabel("SHAP contribution to the predicted-class probability")
    records.append(dict(alert_role=title, true_class=yte.iloc[i], predicted_class=str(pred[i]),
                        predicted_probability=round(float(proba[i].max()), 3),
                        predicted_class_is_normal=str(pred[i]) in r.NORMAL,
                        top_contributions=[dict(feature=cols[j], source=r.group_of(cols[j]), value=float(x.iloc[0, j]),
                                                shap=round(float(contrib[j]), 4)) for j in o[:5]]))
fig.tight_layout(); fig.savefig(FIG / "fig7_local_shap_alerts.png", dpi=300); plt.close(fig)
json.dump(records, open(TAB / "example_alert_records.json", "w"), indent=2)

# ---- probability calibration of the compact model's maximum class probability (seed 0)
conf = proba.max(axis=1); corr = (pred == yte.values)
bins = np.array([0, 0.5, 0.7, 0.9, 0.99, 1.0001]); rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel_ = (conf >= lo) & (conf < hi)
    if sel_.sum(): rows.append(dict(confidence_bin=f"[{lo:.2f}, {min(hi, 1):.2f})", flows=int(sel_.sum()), accuracy=float(corr[sel_].mean())))
pd.DataFrame(rows).to_csv(TAB / "confidence_bins_seed0.csv", index=False)
print("figures written to", FIG)
