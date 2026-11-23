#!/usr/bin/env python3
"""
SNIFF - train the TinyML classifier with TensorFlow and export it as a
quantized TensorFlow Lite model for TensorFlow Lite Micro on the ESP32.

Author: Ayaan Khan

This is the path described in Figure 1 of the proposal: a TinyML model in
TF Lite form, doing binary classification between Safe and Gas Leak, with
inference running locally on the microcontroller.

    pip install tensorflow numpy pandas scikit-learn matplotlib
    python3 train_tflite.py

Outputs:
    firmware/sniff_web/model_data.h    int8 .tflite flatbuffer as a C array
    results/metrics_tflite.json
    results/fig1_humidity_effect.png
    results/fig2_false_alarms.png
"""

import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

RNG = 42
FEATURES = ["rs_ro", "temp_c", "rh_pct", "drs_dt"]
DATA_DIR, OUT_DIR = "data", "results"
HEADER = os.path.join("firmware", "sniff_web", "model_data.h")

np.random.seed(RNG)
tf.random.set_seed(RNG)


# ----------------------------------------------------------------- data

def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSVs in {DATA_DIR}/. Record sessions with collect.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    need = FEATURES + ["label", "session_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"CSVs missing columns: {miss}")
    df = df.dropna(subset=FEATURES + ["label"])
    df = df[(df["rs_ro"] > 0) & np.isfinite(df["rs_ro"])]
    print(f"Loaded {len(files)} sessions, {len(df)} usable rows.\n")
    return df


def provenance(df):
    sid = df["session_id"].astype(str)
    note = df.get("notes", pd.Series([""] * len(df))).astype(str)
    sim = sid.str.startswith("SYNTH_") | note.str.contains("SYNTHETIC|simulated", case=False)
    n_s, n_r = int(sim.sum()), int((~sim).sum())
    if n_s and not n_r:
        print("Data source: SIMULATED (make_synthetic.py). Label the metrics")
        print("below as simulated wherever they are reported.\n")
        return "simulated"
    if n_s and n_r:
        print(f"MIXED: {n_r} measured and {n_s} simulated rows. Metrics over a")
        print("mix describe neither. Run make_synthetic.py --clean first.\n")
        return "MIXED"
    return "measured"


def split(df, frac=0.3):
    rng = np.random.default_rng(RNG)
    tr_s, te_s = [], []
    for _, g in df.groupby(["rh_condition", "label"]):
        s = sorted(g["session_id"].unique())
        if len(s) == 1:
            tr_s += s
        else:
            k = max(1, int(round(len(s) * frac)))
            picked = list(rng.choice(s, size=k, replace=False))
            te_s += picked
            tr_s += [x for x in s if x not in picked]
    if not te_s:
        print("Only one session per condition: falling back to a time-based split.")
        tr, te = [], []
        for _, g in df.groupby("session_id"):
            g = g.sort_values("millis"); c = int(len(g) * 0.7)
            tr.append(g.iloc[:c]); te.append(g.iloc[c:])
        return pd.concat(tr), pd.concat(te), "time-based (fallback)"
    print(f"Split: {len(tr_s)} train sessions, {len(te_s)} held-out sessions\n")
    return df[df.session_id.isin(tr_s)], df[df.session_id.isin(te_s)], "session-wise"


# ----------------------------------------------------------------- metrics

def ev(name, y, p, te, res):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = ((te["label"] == "safe") & (te["rh_condition"] == "humid")).values
    res[name] = dict(
        accuracy=(tp + tn) / len(y),
        recall=tp / (tp + fn) if (tp + fn) else float("nan"),
        precision=tp / (tp + fp) if (tp + fp) else float("nan"),
        fpr=fp / (fp + tn) if (fp + tn) else float("nan"),
        humid_fpr=float(np.mean(np.asarray(p)[m] == 1)) if m.sum() else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return res[name]


def table(res):
    print("\nResults on held-out data")
    print("=" * 86)
    print(f"{'method':<34s} {'acc':>7s} {'recall':>8s} {'prec':>7s} {'FPR':>7s} {'FPR humid/no-gas':>18s}")
    print("-" * 86)
    for k, v in res.items():
        hf = "n/a" if np.isnan(v["humid_fpr"]) else f"{v['humid_fpr']*100:6.1f}%"
        print(f"{k:<34s} {v['accuracy']*100:6.1f}% {v['recall']*100:7.1f}% "
              f"{v['precision']*100:6.1f}% {v['fpr']*100:6.1f}% {hf:>18s}")
    print("=" * 86)


def tune(scores, y):
    """Pick the fixed-threshold alarm point, fairly.

    This searches a dense LINEAR grid, not quantiles. Quantiles follow the
    data density, and this dataset is strongly bimodal: leak rows cluster
    near Rs/Ro 0.1 and safe rows near 1.6, with an empty gap between them.
    Quantiles pile up inside the two clusters and almost none land in the
    gap, so a quantile search puts the threshold hard against the lower
    edge of the safe cluster. That threshold separates the training data
    perfectly and then false-alarms on anything that drifts slightly low,
    which makes the baseline look far worse than a fixed threshold really
    is. Comparing a tuned model against a strawman baseline is not a result.

    Where several thresholds tie on balanced accuracy, this takes the
    MIDDLE of the widest tying run, which is the most robust choice and
    the one an engineer setting the trip point by hand would make.
    """
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return lo
    grid = np.linspace(lo, hi, 2000)
    best_s, scored = -1.0, []
    for t in grid:
        p = (scores < t).astype(int)
        tp = ((p == 1) & (y == 1)).sum(); fn = ((p == 0) & (y == 1)).sum()
        tn = ((p == 0) & (y == 0)).sum(); fp = ((p == 1) & (y == 0)).sum()
        s = ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) / 2
        scored.append(s)
        if s > best_s: best_s = s
    scored = np.asarray(scored)
    tied = np.flatnonzero(scored >= best_s - 1e-12)

    # widest contiguous run of tying thresholds, then its midpoint
    runs, start = [], tied[0]
    for a, b in zip(tied[:-1], tied[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, tied[-1]))
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    return float(grid[(i0 + i1) // 2])


# ----------------------------------------------------------------- main

def main():
    df = load()
    src = provenance(df)
    if df["label"].nunique() < 2:
        sys.exit("Only one class present. Record both safe and leak sessions.")

    tr, te, kind = split(df)
    res = {}
    ytr = (tr["label"] == "leak").astype(int).values
    yte = (te["label"] == "leak").astype(int).values

    # --- conventional baselines -------------------------------------
    print("Baselines")
    print("-" * 62)
    t0 = tune(tr["rs_ro"].values, ytr)
    ev("fixed threshold", yte, (te["rs_ro"].values < t0).astype(int), te, res)
    print(f"Fixed threshold: alarm when Rs/Ro < {t0:.4f}")

    safe = tr[tr.label == "safe"]
    comp = None
    if len(safe) >= 20:
        comp = LinearRegression().fit(safe[["temp_c", "rh_pct"]].values, safe["rs_ro"].values)
        rtr = tr["rs_ro"].values - comp.predict(tr[["temp_c", "rh_pct"]].values)
        rte = te["rs_ro"].values - comp.predict(te[["temp_c", "rh_pct"]].values)
        t1 = tune(rtr, ytr)
        ev("threshold + T/RH compensation", yte, (rte < t1).astype(int), te, res)
        print(f"Compensated: Rs/Ro = {comp.intercept_:.4f} {comp.coef_[0]:+.5f}*T "
              f"{comp.coef_[1]:+.5f}*RH, alarm on residual < {t1:.4f}")

    # --- Keras model -------------------------------------------------
    print("\nTinyML model (TensorFlow / Keras)")
    print("-" * 62)
    Xtr = tr[FEATURES].values.astype(np.float32)
    Xte = te[FEATURES].values.astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    # Validation is split by SESSION, never by row.
    #
    # Keras' validation_split shuffles rows. At 1 Hz, consecutive samples are
    # near-duplicates, so a row split puts copies of the training data into
    # validation, the validation loss tracks the training loss, and early
    # stopping never fires. That is how an unregularised net on 16 sessions
    # reached 100% on its own validation and 55% on held-out sessions.
    #
    # Regularisation is sized for the real sample count. There are 16 sessions
    # here, not 4840 independent observations; the effective sample size is
    # closer to the number of sessions than to the number of rows, because
    # within a session the sensor state is highly autocorrelated.
    tr_sessions = sorted(tr["session_id"].unique())
    n_val = max(1, len(tr_sessions) // 6)
    val_sessions = tr_sessions[1::max(1, len(tr_sessions) // n_val)][:n_val]
    vmask = tr["session_id"].isin(val_sessions).values
    print(f"Validation sessions (held out of the fit): {list(val_sessions)}")

    Zfit, yfit = Ztr[~vmask], ytr[~vmask]
    Zval, yval = Ztr[vmask],  ytr[vmask]

    reg = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(Zfit, yfit, epochs=300, batch_size=32, verbose=0,
                     validation_data=(Zval, yval),
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                         patience=25, restore_best_weights=True,
                         monitor="val_loss")])
    print(f"Stopped after {len(hist.history['loss'])} epochs "
          f"(best val_loss {min(hist.history['val_loss']):.4f}).")

    n_par = int(sum(np.prod(w.shape) for w in model.get_weights()))
    print(f"Trained. {n_par} parameters.")
    ev("TinyML float32 (Keras)", yte, np.argmax(model.predict(Zte, verbose=0), 1), te, res)

    # --- TFLite int8 conversion --------------------------------------
    print("\nTensorFlow Lite conversion")
    print("-" * 62)

    def rep():
        idx = np.random.default_rng(RNG).choice(len(Ztr), min(500, len(Ztr)), replace=False)
        for i in idx:
            yield [Ztr[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    print(f"int8 .tflite: {len(blob)} bytes")

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    i_s, i_z = ind["quantization"]
    o_s, o_z = outd["quantization"]
    print(f"input  quant: scale {i_s:.8f} zero_point {i_z}")
    print(f"output quant: scale {o_s:.8f} zero_point {o_z}")

    preds = []
    for row in Zte:
        q = np.clip(np.round(row / i_s + i_z), -128, 127).astype(np.int8)
        interp.set_tensor(ind["index"], q.reshape(1, -1))
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(outd["index"])[0])))
    ev("TinyML int8 (TF Lite)", yte, np.array(preds), te, res)

    d = res["TinyML int8 (TF Lite)"]["accuracy"] - res["TinyML float32 (Keras)"]["accuracy"]
    print(f"Quantization cost: {d*100:+.2f} percentage points")

    # --- model for DEPLOYMENT ----------------------------------------
    #
    # The model evaluated above was fitted on the training split, so its
    # held-out score is honest. The model SHIPPED to the device is refitted
    # on all sessions, because a deployed detector should use every
    # measurement available. Nothing here is selected on held-out score; the
    # recipe (architecture, regularisation, early stopping, session-level
    # validation) is identical, only the data it sees is larger.
    #
    # Report the split numbers. Ship this one.
    print("\nRefitting on all sessions for deployment")
    print("-" * 62)
    all_sessions = sorted(df["session_id"].unique())
    dep_val = all_sessions[1::max(1, len(all_sessions) // 2)][:2]
    dvm = df["session_id"].isin(dep_val).values
    Xall = df[FEATURES].values.astype(np.float32)
    mu, sd = Xall[~dvm].mean(0), Xall[~dvm].std(0)
    sd[sd < 1e-9] = 1.0
    Zall = (Xall - mu) / sd
    yall = (df["label"] == "leak").astype(int).values
    print(f"Validation sessions held out of the deployment fit: {list(dep_val)}")

    reg2 = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg2),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(Zall[~dvm], yall[~dvm], epochs=300, batch_size=32, verbose=0,
                   validation_data=(Zall[dvm], yall[dvm]),
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       patience=25, restore_best_weights=True, monitor="val_loss")])
    fit_acc = (np.argmax(model.predict(Zall, verbose=0), 1) == yall).mean()
    print(f"Stopped after {len(h2.history['loss'])} epochs. "
          f"Accuracy on all data it was fitted on: {fit_acc*100:.1f}% "
          f"(NOT a held-out figure, do not report it).")
    def rep_dep():
        idx = np.random.default_rng(RNG).choice(len(Zall), min(500, len(Zall)), replace=False)
        for i in idx:
            yield [Zall[i:i+1].astype(np.float32)]

    conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv2.optimizations = [tf.lite.Optimize.DEFAULT]
    conv2.representative_dataset = rep_dep
    conv2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv2.inference_input_type = tf.int8
    conv2.inference_output_type = tf.int8
    blob = conv2.convert()
    print(f"Deployment int8 .tflite: {len(blob)} bytes  <- this is what ships")


    table(res)

    base = res.get("threshold + T/RH compensation") or res["fixed threshold"]
    best = res["TinyML int8 (TF Lite)"]
    if not (np.isnan(base["humid_fpr"]) or np.isnan(best["humid_fpr"])):
        dl = base["humid_fpr"] - best["humid_fpr"]
        print(f"VERDICT: the model changes the humid-air false alarm rate by "
              f"{-dl*100:+.1f} points against the best baseline.")
        if dl <= 0:
            print("         It did NOT beat the baseline. Report that; it is a result.")

    # --- figures ------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    s = df[df.label == "safe"]
    if len(s) > 10 and s["rh_pct"].nunique() > 2:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.scatter(s["rh_pct"], s["rs_ro"], s=6, alpha=.35, color="#2B5578", edgecolors="none")
        z = np.polyfit(s["rh_pct"], s["rs_ro"], 1)
        xs = np.linspace(s["rh_pct"].min(), s["rh_pct"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#C04A2E", lw=2, label=f"slope {z[0]:+.4f} per %RH")
        ax.legend(frameon=False); ax.set_xlabel("Relative humidity (%)"); ax.set_ylabel("Rs / Ro")
        ax.set_title("Clean air only: resistance falls as humidity rises")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig1_humidity_effect.png", dpi=160); plt.close(fig)
        print(f"\n  {OUT_DIR}/fig1_humidity_effect.png")

    names = [k for k in res if not np.isnan(res[k]["humid_fpr"])]
    if names:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        vals = [res[k]["humid_fpr"] * 100 for k in names]
        ax.barh(names, vals, color=["#C04A2E" if "threshold" in k else "#2F7A52" for k in names])
        for i, v in enumerate(vals): ax.text(v + .6, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("False alarm rate in humid air with no gas (%)")
        ax.set_title("The comparison that matters")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig2_false_alarms.png", dpi=160); plt.close(fig)
        print(f"  {OUT_DIR}/fig2_false_alarms.png")

    # --- deployment constants for the conventional detectors ----------
    #
    # The thresholds REPORTED above are fitted on the training split only, so
    # the held-out comparison stays honest. The thresholds DEPLOYED to the
    # firmware are refitted on the whole dataset, because a shipped detector
    # should use everything that was measured. They are written into
    # model_data.h so the firmware can never drift out of sync with training.
    y_all = (df["label"] == "leak").astype(int).values
    dep_t0 = tune(df["rs_ro"].values, y_all)
    safe_all = df[df.label == "safe"]
    dep_a = dep_b = dep_c = 0.0
    dep_t1 = 0.0
    if len(safe_all) >= 20:
        c_all = LinearRegression().fit(safe_all[["temp_c", "rh_pct"]].values,
                                       safe_all["rs_ro"].values)
        res_all = df["rs_ro"].values - c_all.predict(df[["temp_c", "rh_pct"]].values)
        dep_t1 = tune(res_all, y_all)
        dep_a, dep_b, dep_c = float(c_all.intercept_), float(c_all.coef_[0]), float(c_all.coef_[1])
    print(f"\nDeployment constants (full-dataset fit, written to model_data.h):")
    print(f"  fixed      : alarm when Rs/Ro < {dep_t0:.4f}")
    print(f"  compensated: expected = {dep_a:.4f} {dep_b:+.5f}*T {dep_c:+.5f}*RH")
    print(f"               alarm when residual < {dep_t1:+.4f}")

    # --- C array ------------------------------------------------------
    os.makedirs(os.path.dirname(HEADER), exist_ok=True)
    body = ",\n  ".join(", ".join(f"0x{b:02x}" for b in blob[i:i+12])
                        for i in range(0, len(blob), 12))
    with open(HEADER, "w") as f:
        f.write(f"""// SNIFF - TensorFlow Lite model. GENERATED BY train_tflite.py, DO NOT EDIT.
//
// Author: Ayaan Khan
//
// Architecture:   4 -> 16 -> 8 -> 2 (ReLU, softmax)
// Parameters:     {n_par}
// Flatbuffer:     {len(blob)} bytes, full int8 quantization
// Data source:    {src}
// Held-out:       accuracy {best['accuracy']*100:.1f}%, leak recall {best['recall']*100:.1f}%
//                 (measured on the SPLIT-trained model; see below)
//
// WEIGHTS BELOW ARE THE DEPLOYMENT REFIT, trained on all 16 sessions with the
// identical recipe. They have seen every session, so they have no held-out
// score of their own. Report the split numbers above, never a figure measured
// on these weights.
// Quantization:   {d*100:+.2f} percentage points vs float32
//
// Feature order: rs_ro, temp_c, rh_pct, drs_dt
// Normalisation is frozen here and MUST match training exactly.

#ifndef SNIFF_MODEL_DATA_H
#define SNIFF_MODEL_DATA_H

#define SNIFF_N_FEATURES {len(FEATURES)}

static const float SNIFF_FEAT_MEAN[{len(FEATURES)}] = {{{', '.join(f'{v:.8f}f' for v in mu)}}};
static const float SNIFF_FEAT_STD[{len(FEATURES)}]  = {{{', '.join(f'{v:.8f}f' for v in sd)}}};

// ---- conventional detectors, for the live side-by-side comparison --------
// Refitted on the full dataset. See train_tflite.py for why these differ
// slightly from the thresholds in the held-out results table.
//
// Detector 1, fixed threshold:   alarm when rs_ro < SNIFF_FIXED_T
// Detector 2, T/RH compensated:  expected = A + B*temp_c + C*rh_pct
//                                alarm when (rs_ro - expected) < SNIFF_COMP_T
#define SNIFF_FIXED_T  {dep_t0:.6f}f
#define SNIFF_COMP_A   {dep_a:.6f}f
#define SNIFF_COMP_B   {dep_b:.8f}f
#define SNIFF_COMP_C   {dep_c:.8f}f
#define SNIFF_COMP_T   {dep_t1:.6f}f

alignas(16) static const unsigned char sniff_model_tflite[] = {{
  {body}
}};
static const unsigned int sniff_model_tflite_len = {len(blob)};

#endif  // SNIFF_MODEL_DATA_H
""")
    print(f"  {HEADER}  ({len(blob)} bytes)")

    with open(f"{OUT_DIR}/metrics_tflite.json", "w") as f:
        json.dump(dict(data_source=src, split=kind, parameters=n_par,
                       tflite_bytes=len(blob),
                       input_quant=dict(scale=float(i_s), zero_point=int(i_z)),
                       output_quant=dict(scale=float(o_s), zero_point=int(o_z)),
                       results=res), f, indent=2)
    print(f"  {OUT_DIR}/metrics_tflite.json")
    print("\nNext: flash firmware/sniff_web")


if __name__ == "__main__":
    main()
