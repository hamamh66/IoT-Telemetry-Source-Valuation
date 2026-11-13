"""
rtiot_xai.py -- Compact telemetry and explainable alerts for IoT security monitoring on RT-IoT2022.

Experiments (all with stratified train/test splits; every fitted step uses training data only):
  * audit      -- effect of the row-index column, of a regressor-with-rounding instead of a classifier,
                  and of SHAP-based selection, on macro-F1;
  * compact    -- two-pass classifier (SHAP top-k selection on the training split, then refit) against the
                  full 83-feature forest, impurity top-k and random k-feature baselines;
  * streams    -- exact Shapley attribution of macro-F1 to five telemetry sources (all 32 coalitions);
  * degraded   -- loss of performance when one telemetry source is unavailable at inference time.

Dataset: RT-IoT2022 (Sharmila and Nagapadma, Cybersecurity 6, 2023; UCI Machine Learning Repository,
dataset 942, CC BY 4.0). The CSV's first column ('Unnamed: 0') is a per-class row counter written when the
file was exported; it is not a traffic feature.
"""
from __future__ import annotations
import itertools, math, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix  # noqa: F401

SEEDS = [0, 1, 2, 3, 4]          # repeated stratified 60/40 splits
TEST_SIZE = 0.40
N_TREES = 100
K_SELECT = 20                     # size of the compact feature set
SHAP_SAMPLE = 1500                # training rows used to rank features with TreeSHAP
INDEX_COL = "Unnamed: 0"
LABEL = "Attack_type"
# Thing_Speak, Wipro_bulb and MQTT_Publish are normal device traffic in RT-IoT2022
NORMAL = {"Thing_Speak", "Wipro_bulb", "MQTT_Publish"}

def group_of(col: str) -> str:
    """Telemetry source that must be collected and shipped to compute a feature."""
    if col in ("id.orig_p", "id.resp_p"):
        return "endpoint"
    if col in ("proto", "service"):
        return "protocol"
    if "header_size" in col or "flag_count" in col or "window_size" in col:
        return "header"
    if ("iat" in col or col.startswith("active") or col.startswith("idle") or col == "flow_duration"
            or col.endswith("per_sec") or col == "payload_bytes_per_second"):
        return "timing"
    return "volume"          # packet counts, payload statistics, subflow, bulk, down/up ratio

GROUPS = ["timing", "volume", "header", "protocol", "endpoint"]


def load(path: str, keep_index: bool = False):
    d = pd.read_csv(path)
    y = d[LABEL].astype(str)
    X = d.drop(columns=[LABEL] + ([] if keep_index else [INDEX_COL]))
    for c in ("proto", "service"):
        X[c] = X[c].astype("category").cat.codes
    return X, y


def split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


def rf(seed, n_jobs=-1):
    return RandomForestClassifier(n_estimators=N_TREES, n_jobs=n_jobs, random_state=seed)


def shap_rank(model, Xtr, seed, regression=False, ytr=None):
    """Mean |TreeSHAP| over a training subsample (and over classes for a classifier)."""
    import shap
    rs = np.random.default_rng(seed)
    if ytr is None:
        idx = rs.choice(len(Xtr), size=min(SHAP_SAMPLE, len(Xtr)), replace=False)
    else:   # class-balanced sample: an equal share of the SHAP sample for every class (all rows of rare classes)
        per = SHAP_SAMPLE // ytr.nunique(); yv = ytr.values; idx = []
        for c in np.unique(yv):
            pos = np.flatnonzero(yv == c); idx += list(rs.choice(pos, size=min(per, len(pos)), replace=False))
        idx = np.array(idx)
    sv = shap.TreeExplainer(model).shap_values(Xtr.iloc[idx], check_additivity=False)
    sv = np.asarray(sv)
    if regression:
        imp = np.abs(sv).mean(axis=0)
    else:   # (n, features, classes) in shap >= 0.45; older versions return a list over classes
        if sv.ndim == 3 and sv.shape[0] == len(idx):
            imp = np.abs(sv).mean(axis=(0, 2))
        else:
            imp = np.abs(sv).mean(axis=(0, 1))
    return pd.Series(imp, index=Xtr.columns).sort_values(ascending=False)


def metrics(y_true, y_pred, classes):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    yt_att = ~pd.Series(y_true).isin(NORMAL).values
    yp_att = ~pd.Series(y_pred).isin(NORMAL).values
    tp = (yt_att & yp_att).sum(); fp = (~yt_att & yp_att).sum(); fn = (yt_att & ~yp_att).sum()
    return dict(macro_f1=f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                accuracy=float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
                attack_recall=tp / max(tp + fn, 1), attack_precision=tp / max(tp + fp, 1),
                per_class=pd.DataFrame(dict(precision=p, recall=r, f1=f, support=s), index=classes))


# --------------------------------------------------------------------------- A. leakage audit
def audit(path, seed, verbose=True):
    rows = []
    for keep_index in (True, False):
        X, y = load(path, keep_index=keep_index)
        classes = sorted(y.unique())
        code = {c: i for i, c in enumerate(classes)}
        Xtr, Xte, ytr, yte = split(X, y, seed)
        # regressor on the integer class codes, top-20 by SHAP, refit, round and clip
        reg = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=seed).fit(Xtr, ytr.map(code))
        top = shap_rank(reg, Xtr, seed, regression=True).index[:K_SELECT]
        reg2 = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=seed).fit(Xtr[top], ytr.map(code))
        pr = np.clip(np.rint(reg2.predict(Xte[top])), 0, len(classes) - 1).astype(int)
        m = metrics(yte.values, np.array(classes)[pr], classes)
        rows.append(dict(seed=seed, index_column=keep_index, model="regressor + rounding, SHAP top-20",
                         index_in_top20=INDEX_COL in top, **{k: v for k, v in m.items() if k != "per_class"}))
        # classifier, SHAP top-20
        clf = rf(seed).fit(Xtr, ytr)
        top = shap_rank(clf, Xtr, seed).index[:K_SELECT]
        clf2 = rf(seed).fit(Xtr[top], ytr)
        m = metrics(yte.values, clf2.predict(Xte[top]), classes)
        rows.append(dict(seed=seed, index_column=keep_index, model="classifier, SHAP top-20",
                         index_in_top20=INDEX_COL in top, **{k: v for k, v in m.items() if k != "per_class"}))
        if verbose: print(f"  audit seed={seed} index_column={keep_index} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- compact model and baselines
def compact(path, seed, verbose=True):
    X, y = load(path); classes = sorted(y.unique())
    Xtr, Xte, ytr, yte = split(X, y, seed)
    rows, per_class, preds = [], [], {}
    t = time.time(); full = rf(seed).fit(Xtr, ytr); fit_full = time.time() - t
    rank = shap_rank(full, Xtr, seed, ytr=ytr)                 # class-balanced SHAP sample
    rank_unbal = shap_rank(full, Xtr, seed)                     # sample drawn at the class frequencies
    imp = pd.Series(full.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    rs = np.random.default_rng(1000 + seed)
    sets = {"full (83 features)": list(Xtr.columns),
            "two-pass, SHAP top-20": list(rank.index[:K_SELECT]),
            "two-pass, SHAP top-20 (unbalanced sample)": list(rank_unbal.index[:K_SELECT]),
            "impurity top-20": list(imp.index[:K_SELECT])}
    for r in range(3):
        sets[f"random 20 (draw {r + 1})"] = list(rs.choice(Xtr.columns, K_SELECT, replace=False))
    models = {}
    for name, cols in sets.items():
        if name.startswith("full"):
            m_ = full; ft = fit_full
        else:
            t = time.time(); m_ = rf(seed).fit(Xtr[cols], ytr); ft = time.time() - t
        t = time.time(); p = m_.predict(Xte[cols]); it = (time.time() - t) / len(Xte) * 1e6
        mm = metrics(yte.values, p, classes)
        rows.append(dict(seed=seed, model=name, n_features=len(cols),
                         n_groups=len({group_of(c) for c in cols}), fit_s=ft, infer_us_per_flow=it,
                         **{k: v for k, v in mm.items() if k != "per_class"}))
        pc = mm["per_class"].assign(seed=seed, model=name); per_class.append(pc.reset_index(names="class"))
        models[name] = (m_, cols); preds[name] = p
    selected = pd.DataFrame(dict(seed=seed, rank=np.arange(1, K_SELECT + 1), feature=rank.index[:K_SELECT],
                                 mean_abs_shap=rank.values[:K_SELECT],
                                 group=[group_of(c) for c in rank.index[:K_SELECT]]))
    cm = confusion_matrix(yte.values, preds["two-pass, SHAP top-20"], labels=classes)
    if verbose: print(f"  compact seed={seed} done", flush=True)
    return pd.DataFrame(rows), pd.concat(per_class), selected, cm, models, (Xtr, Xte, ytr, yte, classes)


# --------------------------------------------------------------------------- B. value of telemetry streams
def coalitions(path, seed, verbose=True):
    X, y = load(path); classes = sorted(y.unique())
    Xtr, Xte, ytr, yte = split(X, y, seed)
    gcol = {g: [c for c in X.columns if group_of(c) == g] for g in GROUPS}
    rows = []
    for r in range(len(GROUPS) + 1):
        for S in itertools.combinations(GROUPS, r):
            if not S:   # no telemetry: predict the majority class of the training split
                p = np.full(len(yte), ytr.value_counts().index[0])
            else:
                cols = sum((gcol[g] for g in S), [])
                p = rf(seed).fit(Xtr[cols], ytr).predict(Xte[cols])
            mm = metrics(yte.values, p, classes)
            rows.append(dict(seed=seed, coalition="+".join(S) or "none",
                             **{k: v for k, v in mm.items() if k != "per_class"}))
    if verbose: print(f"  coalitions seed={seed} done", flush=True)
    return pd.DataFrame(rows)


def shapley(df, metric="macro_f1"):
    from scipy import stats
    n = len(GROUPS); key = lambda S: "+".join(g for g in GROUPS if g in S) or "none"
    v = df.pivot(index="seed", columns="coalition", values=metric); out = []
    phis = {}
    for g in GROUPS:
        others = [x for x in GROUPS if x != g]; acc = 0.0
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                acc = acc + w * (v[key(S + (g,))] - v[key(S)])
        phis[g] = acc
    resid = float(np.abs(sum(phis.values()) - (v[key(GROUPS)] - v["none"])).max())
    T = len(v); tc = stats.t.ppf(0.975, T - 1)
    for g in GROUPS:
        x = phis[g].values; h = tc * x.std(ddof=1) / np.sqrt(T)
        alone = (v[key((g,))] - v["none"]).mean()
        last = (v[key(GROUPS)] - v[key(tuple(o for o in GROUPS if o != g))]).mean()
        out.append(dict(metric=metric, group=g, shapley=x.mean(), ci_low=x.mean() - h, ci_high=x.mean() + h,
                        alone=alone, added_last=last, efficiency_residual=resid))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- C. degraded telemetry
def degraded(models, data, seed, verbose=True):
    Xtr, Xte, ytr, yte, classes = data
    rows, per_class = [], []
    for name in ("full (83 features)", "two-pass, SHAP top-20"):
        m_, cols = models[name]
        fill = Xtr[cols].median()
        for g in ["none"] + GROUPS:
            Z = Xte[cols].copy()
            lost = [c for c in cols if group_of(c) == g]
            if lost: Z[lost] = fill[lost].values          # source unavailable: values replaced by training median
            mm = metrics(yte.values, m_.predict(Z), classes)
            rows.append(dict(seed=seed, model=name, missing_source=g, n_lost=len(lost),
                             **{k: v for k, v in mm.items() if k != "per_class"}))
            per_class.append(mm["per_class"].assign(seed=seed, model=name, missing_source=g).reset_index(names="class"))
    if verbose: print(f"  degraded seed={seed} done", flush=True)
    return pd.DataFrame(rows), pd.concat(per_class)


def ci(x):
    from scipy import stats
    x = np.asarray(x, float); n = len(x); m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, m - h, m + h
