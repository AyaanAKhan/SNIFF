#!/usr/bin/env python3
"""
SNIFF - train the classifier, compare it against the conventional detector,
and export the weights for the ESP32.

Author: Ayaan Khan

Run this after collecting sessions with collect.py.

    pip install numpy pandas scikit-learn matplotlib
    python train.py

What it does, in order:
  1. Loads every CSV in data/ and reports what conditions you actually have.
  2. Splits by SESSION, never by row. Samples one second apart are near
     duplicates; a random row split would put them in both train and test
     and hand you a meaningless accuracy figure.
  3. Trains two models: logistic regression (floor) and a small MLP.
  4. Builds two conventional baselines: a fixed threshold, and a threshold
     compensated for temperature and humidity. The comparison against these
     is the actual contribution of the project.
  5. Measures what int8 weight quantization costs in accuracy.
  6. Writes firmware/sniff_inference/model.h and the figures.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

RNG = 42
FEATURES = ["rs_ro", "temp_c", "rh_pct", "drs_dt"]
DATA_DIR = "data"
OUT_DIR = "results"
HEADER_PATH = os.path.join("firmware", "sniff_inference", "model.h")


# ----------------------------------------------------------------- loading

def load_sessions():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSVs in {DATA_DIR}/. Record sessions with collect.py first.")

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  skipping {f}: {e}")
            continue
        if d.empty:
            print(f"  skipping {f}: empty")
            continue
        frames.append(d)

    if not frames:
        sys.exit("Every CSV was empty or unreadable.")

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in FEATURES + ["label", "session_id"] if c not in df.columns]
    if missing:
        sys.exit(f"CSVs are missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=FEATURES + ["label"])
    # Rows before Ro calibration have rs_ro = 0 or nonsense; drop them.
    df = df[(df["rs_ro"] > 0) & np.isfinite(df["rs_ro"])]
    dropped = before - len(df)

    print(f"Loaded {len(files)} session files, {len(df)} usable rows "
          f"({dropped} dropped for missing or pre-calibration values).\n")
    return df


def data_source(df):
    """Report whether the dataset is measured, simulated, or a mix of both.
    A mix is the dangerous case: the numbers then describe neither."""
    sid = df["session_id"].astype(str)
    note = df.get("notes", pd.Series([""] * len(df))).astype(str)
    synth = sid.str.startswith("SYNTH_") | note.str.contains("SYNTHETIC", case=False)
    n_s, n_r = int(synth.sum()), int((~synth).sum())

    if n_s and not n_r:
        print("Data source: SIMULATED. These rows come from make_synthetic.py,")
        print("not from the sensor. The metrics below describe how well the model")
        print("fits the generator's equations, so label them as simulated wherever")
        print("they are reported.\n")
        return "simulated"

    if n_s and n_r:
        print("!" * 70)
        print(f"MIXED: {n_r} measured rows and {n_s} simulated rows.")
        print("Metrics over a mix describe neither one. Keep them separate:")
        print("  python make_synthetic.py --clean     (keep only measured data)")
        print("!" * 70 + "\n")
        return "MIXED"

    return "measured"


def describe(df):
    print("Conditions recorded")
    print("-" * 62)
    g = (df.groupby(["rh_condition", "label"])
           .agg(rows=("rs_ro", "size"),
                sessions=("session_id", "nunique"),
                mean_rh=("rh_pct", "mean"),
                mean_rs_ro=("rs_ro", "mean"))
           .reset_index())
    for _, r in g.iterrows():
        print(f"  {r['rh_condition']:<8s} {r['label']:<5s}  "
              f"{int(r['rows']):5d} rows  {int(r['sessions'])} session(s)  "
              f"RH {r['mean_rh']:5.1f}%  mean Rs/Ro {r['mean_rs_ro']:.3f}")
    print()

    # The cell the whole premise rests on.
    critical = df[(df["label"] == "safe") & (df["rh_condition"] == "humid")]
    if critical.empty:
        print("WARNING: you have no humid + no-gas data.")
        print("That is the false-alarm case the entire project is about.")
        print("Without it the baseline comparison cannot show anything.\n")
    else:
        print(f"Critical cell present: {len(critical)} rows of humid air with no gas.\n")

    return g


# ----------------------------------------------------------------- splitting

def split_by_session(df, holdout_frac=0.3):
    """Hold out whole sessions, balanced across conditions where possible."""
    rows = []
    rng = np.random.default_rng(RNG)
    for (cond, lab), grp in df.groupby(["rh_condition", "label"]):
        sess = sorted(grp["session_id"].unique())
        if len(sess) == 1:
            rows.append((cond, lab, sess, []))      # cannot hold out; goes to train
        else:
            n_hold = max(1, int(round(len(sess) * holdout_frac)))
            picked = list(rng.choice(sess, size=n_hold, replace=False))
            rows.append((cond, lab, [s for s in sess if s not in picked], picked))

    train_s = [s for r in rows for s in r[2]]
    test_s = [s for r in rows for s in r[3]]

    if not test_s:
        print("WARNING: only one session per condition, so no session can be held out.")
        print("Falling back to a time-based split: last 30% of each session is the")
        print("test set. This is weaker than a session split. Record a second run")
        print("of each condition if you have time, and rerun.\n")
        tr, te = [], []
        for _, grp in df.groupby("session_id"):
            grp = grp.sort_values("millis")
            cut = int(len(grp) * 0.7)
            tr.append(grp.iloc[:cut])
            te.append(grp.iloc[cut:])
        return pd.concat(tr), pd.concat(te), "time-based (fallback)"

    train = df[df["session_id"].isin(train_s)]
    test = df[df["session_id"].isin(test_s)]
    print(f"Split: {len(train_s)} training sessions, {len(test_s)} held-out sessions")
    print(f"       {len(train)} train rows, {len(test)} test rows\n")
    return train, test, "session-wise"


# ----------------------------------------------------------------- metrics

def evaluate(name, y_true, y_pred, df_test, results):
    """y: 1 = leak, 0 = safe."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) else float("nan")       # leaks caught
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")          # false alarms
    acc = (tp + tn) / len(y_true)

    # The number that matters: false alarms in humid air with no gas.
    mask = ((df_test["label"] == "safe") & (df_test["rh_condition"] == "humid")).values
    if mask.sum():
        humid_fpr = float(np.mean(np.asarray(y_pred)[mask] == 1))
    else:
        humid_fpr = float("nan")

    results[name] = dict(accuracy=acc, recall=recall, precision=precision,
                         fpr=fpr, humid_fpr=humid_fpr,
                         tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return results[name]


def print_table(results):
    print("\nResults on held-out data")
    print("=" * 86)
    print(f"{'method':<34s} {'acc':>7s} {'recall':>8s} {'prec':>7s} "
          f"{'FPR':>7s} {'FPR humid/no-gas':>18s}")
    print("-" * 86)
    for k, v in results.items():
        hf = "n/a" if np.isnan(v["humid_fpr"]) else f"{v['humid_fpr']*100:6.1f}%"
        print(f"{k:<34s} {v['accuracy']*100:6.1f}% {v['recall']*100:7.1f}% "
              f"{v['precision']*100:6.1f}% {v['fpr']*100:6.1f}% {hf:>18s}")
    print("=" * 86)
    print("recall = leaks caught. FPR humid/no-gas = false alarms in wet air,")
    print("which is the number this whole project exists to reduce.\n")


# ----------------------------------------------------------------- baselines

def fixed_threshold_baseline(train, test, results):
    """Conventional detector: alarm when Rs/Ro drops below a fixed number.
    Threshold chosen on TRAINING data only, picking the value with the best
    balanced accuracy. This is a fair, well-tuned version of the old method."""
    cands = np.quantile(train["rs_ro"], np.linspace(0.02, 0.98, 97))
    ytr = (train["label"] == "leak").astype(int).values
    best, best_score = None, -1
    for t in cands:
        pred = (train["rs_ro"].values < t).astype(int)
        tp = ((pred == 1) & (ytr == 1)).sum(); fn = ((pred == 0) & (ytr == 1)).sum()
        tn = ((pred == 0) & (ytr == 0)).sum(); fp = ((pred == 1) & (ytr == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        if (sens + spec) / 2 > best_score:
            best_score, best = (sens + spec) / 2, t

    pred = (test["rs_ro"].values < best).astype(int)
    evaluate("fixed threshold", (test["label"] == "leak").astype(int).values,
             pred, test, results)
    print(f"Fixed-threshold baseline: alarm when Rs/Ro < {best:.4f}")
    return best


def compensated_threshold_baseline(train, test, results):
    """A stronger baseline: regress Rs/Ro on temperature and humidity using
    only clean-air training rows, then threshold the residual. This is what a
    careful engineer would build without machine learning, and it is the
    honest thing to measure against."""
    safe = train[train["label"] == "safe"]
    if len(safe) < 20:
        print("Not enough clean-air rows to fit the compensated baseline; skipping.")
        return None

    comp = LinearRegression().fit(safe[["temp_c", "rh_pct"]].values, safe["rs_ro"].values)

    res_tr = train["rs_ro"].values - comp.predict(train[["temp_c", "rh_pct"]].values)
    res_te = test["rs_ro"].values - comp.predict(test[["temp_c", "rh_pct"]].values)

    ytr = (train["label"] == "leak").astype(int).values
    cands = np.quantile(res_tr, np.linspace(0.02, 0.98, 97))
    best, best_score = None, -1
    for t in cands:
        pred = (res_tr < t).astype(int)
        tp = ((pred == 1) & (ytr == 1)).sum(); fn = ((pred == 0) & (ytr == 1)).sum()
        tn = ((pred == 0) & (ytr == 0)).sum(); fp = ((pred == 1) & (ytr == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        if (sens + spec) / 2 > best_score:
            best_score, best = (sens + spec) / 2, t

    evaluate("threshold + T/RH compensation", (test["label"] == "leak").astype(int).values,
             (res_te < best).astype(int), test, results)
    print(f"Compensated baseline: Rs/Ro = {comp.intercept_:.4f} "
          f"{comp.coef_[0]:+.5f}*T {comp.coef_[1]:+.5f}*RH, alarm on residual < {best:.4f}")
    return comp


# ----------------------------------------------------------------- quantization

def quantize_int8(w):
    """Symmetric per-tensor int8 quantization. Returns (int8 array, scale)."""
    amax = np.max(np.abs(w))
    if amax == 0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def mlp_forward(X, Ws, bs):
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W + b
        if i < len(Ws) - 1:
            a = np.maximum(a, 0.0)          # relu
    return a


# ----------------------------------------------------------------- figures

def make_figures(df, results):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Figure 1: the premise. Sensor resistance against humidity, clean air only.
    safe = df[df["label"] == "safe"]
    if len(safe) > 10:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.scatter(safe["rh_pct"], safe["rs_ro"], s=6, alpha=0.35,
                   color="#2B5578", edgecolors="none")
        if safe["rh_pct"].nunique() > 2:
            z = np.polyfit(safe["rh_pct"], safe["rs_ro"], 1)
            xs = np.linspace(safe["rh_pct"].min(), safe["rh_pct"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="#C04A2E", lw=2,
                    label=f"slope {z[0]:+.4f} per %RH")
            ax.legend(frameon=False)
        ax.set_xlabel("Relative humidity (%)")
        ax.set_ylabel("Rs / Ro")
        ax.set_title("Clean air only: sensor resistance falls as humidity rises")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig1_humidity_effect.png"), dpi=160)
        plt.close(fig)
        print(f"  {OUT_DIR}/fig1_humidity_effect.png")

    # Figure 2: false alarms in humid air, method by method.
    names = [k for k in results if not np.isnan(results[k]["humid_fpr"])]
    if names:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        vals = [results[k]["humid_fpr"] * 100 for k in names]
        colors = ["#C04A2E" if "threshold" in k else "#2F7A52" for k in names]
        ax.barh(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("False alarm rate in humid air with no gas (%)")
        ax.set_title("The comparison that matters")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig2_false_alarms.png"), dpi=160)
        plt.close(fig)
        print(f"  {OUT_DIR}/fig2_false_alarms.png")


# ----------------------------------------------------------------- export

def write_header(mu, sd, Ws, bs, scales, meta):
    os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)

    def arr2d(W):
        return ",\n  ".join("{" + ", ".join(f"{v:.6f}f" for v in row) + "}" for row in W)

    def arr1d(b):
        return ", ".join(f"{v:.6f}f" for v in b)

    L = [W.shape for W in Ws]
    arch = " -> ".join([str(L[0][0])] + [str(s[1]) for s in L])

    banner = ""
    if meta.get("data_source") != "measured":
        banner = (f"//\n"
                  f"// Data source: {meta['data_source']} "
                  f"(generated by make_synthetic.py, not sensor measurements).\n")

    with open(HEADER_PATH, "w") as f:
        f.write(f"""// SNIFF - trained model weights. GENERATED BY train.py, DO NOT EDIT BY HAND.
{banner}//
// Architecture: {arch}
// Parameters:   {meta['n_params']}
// Data source:  {meta.get('data_source', 'unknown')}
// Trained on:   {meta['n_train']} rows from {meta['n_train_sessions']} session(s)
// Held-out:     accuracy {meta['accuracy']*100:.1f}%, leak recall {meta['recall']*100:.1f}%
// Int8 weight quantization cost: {meta['int8_delta']*100:+.2f} percentage points
//
// Normalisation constants are frozen here and MUST match training exactly.
// Divergence between training and firmware normalisation is the classic
// source of a model that works on the laptop and not on the device.

#ifndef SNIFF_MODEL_H
#define SNIFF_MODEL_H

#define N_FEATURES {len(mu)}
#define N_LAYERS   {len(Ws)}

// Feature order: rs_ro, temp_c, rh_pct, drs_dt
static const float FEAT_MEAN[{len(mu)}] = {{{arr1d(mu)}}};
static const float FEAT_STD[{len(sd)}]  = {{{arr1d(sd)}}};

""")
        for i, (W, b) in enumerate(zip(Ws, bs)):
            f.write(f"// layer {i}: {W.shape[0]} -> {W.shape[1]}  "
                    f"(int8 scale {scales[i]:.8f})\n")
            f.write(f"static const int L{i}_IN = {W.shape[0]};\n")
            f.write(f"static const int L{i}_OUT = {W.shape[1]};\n")
            f.write(f"static const float L{i}_W[{W.shape[0]}][{W.shape[1]}] = {{\n  "
                    f"{arr2d(W)}\n}};\n")
            f.write(f"static const float L{i}_B[{W.shape[1]}] = {{{arr1d(b)}}};\n\n")

        f.write("#endif  // SNIFF_MODEL_H\n")

    print(f"  {HEADER_PATH}")


# ----------------------------------------------------------------- main

def main():
    df = load_sessions()
    source = data_source(df)
    describe(df)

    if df["label"].nunique() < 2:
        sys.exit("Only one class present. You need both leak and safe sessions.")

    train, test, split_kind = split_by_session(df)
    results = {}

    Xtr = train[FEATURES].values.astype(np.float64)
    Xte = test[FEATURES].values.astype(np.float64)
    ytr = (train["label"] == "leak").astype(int).values
    yte = (test["label"] == "leak").astype(int).values

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0                      # a constant feature must not blow up
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    print("Baselines")
    print("-" * 62)
    fixed_threshold_baseline(train, test, results)
    compensated_threshold_baseline(train, test, results)

    print("\nModels")
    print("-" * 62)

    lr = LogisticRegression(max_iter=2000, random_state=RNG).fit(Ztr, ytr)
    evaluate("logistic regression", yte, lr.predict(Zte), test, results)
    print("Logistic regression trained.")

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        max_iter=3000, random_state=RNG, early_stopping=False)
    mlp.fit(Ztr, ytr)
    evaluate("MLP 4-16-8-2", yte, mlp.predict(Zte), test, results)

    Ws = [np.asarray(w, dtype=np.float64) for w in mlp.coefs_]
    bs = [np.asarray(b, dtype=np.float64) for b in mlp.intercepts_]
    n_params = sum(w.size for w in Ws) + sum(b.size for b in bs)
    print(f"MLP trained. {n_params} parameters "
          f"({n_params*4} bytes as float32, {n_params} bytes as int8).")

    # What does int8 weight quantization actually cost?
    qWs, scales = [], []
    for W in Ws:
        q, s = quantize_int8(W)
        qWs.append(q.astype(np.float64) * s)
        scales.append(s)
    logits_f = mlp_forward(Zte, Ws, bs)
    logits_q = mlp_forward(Zte, qWs, bs)
    pred_q = (logits_q[:, 1] > logits_q[:, 0]).astype(int) if logits_q.shape[1] > 1 \
        else (logits_q.ravel() > 0).astype(int)
    evaluate("MLP, int8 weights", yte, pred_q, test, results)

    acc_f = results["MLP 4-16-8-2"]["accuracy"]
    acc_q = results["MLP, int8 weights"]["accuracy"]
    print(f"Int8 weight quantization: {acc_f*100:.1f}% -> {acc_q*100:.1f}% "
          f"({(acc_q-acc_f)*100:+.2f} points)")

    print_table(results)

    # Did the model actually beat the honest baseline?
    base = results.get("threshold + T/RH compensation") or results["fixed threshold"]
    best_model = results["MLP 4-16-8-2"]
    if not np.isnan(base["humid_fpr"]) and not np.isnan(best_model["humid_fpr"]):
        delta = base["humid_fpr"] - best_model["humid_fpr"]
        print(f"VERDICT: the model changes the humid-air false alarm rate by "
              f"{-delta*100:+.1f} percentage points against the best baseline.")
        if delta <= 0:
            print("         The model did NOT beat the baseline. That is a real,")
            print("         reportable result. Say so rather than hiding it.")
    print()

    print("Figures and export")
    print("-" * 62)
    make_figures(df, results)

    meta = dict(n_params=int(n_params), n_train=int(len(train)),
                n_train_sessions=int(train["session_id"].nunique()),
                accuracy=float(acc_f), recall=float(best_model["recall"]),
                int8_delta=float(acc_q - acc_f), split=split_kind,
                data_source=source)
    write_header(mu, sd, Ws, bs, scales, meta)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"data_source": source, "split": split_kind,
                   "meta": meta, "results": results}, f, indent=2)
    print(f"  {OUT_DIR}/metrics.json")

    if source != "measured":
        print(f"\nmodel.h records data_source: {source}.")

    print("\nNext: open firmware/sniff_inference/sniff_inference.ino and flash it.")
    print("model.h is already in that folder.")


if __name__ == "__main__":
    main()
