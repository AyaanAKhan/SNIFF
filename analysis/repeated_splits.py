#!/usr/bin/env python3
"""Repeated session-wise splits.

A single train/test split of 16 sessions is a small sample. Which four
sessions land in the test set changes the answer, and for a 234-parameter
network it changes it a lot. This script repeats the whole evaluation over
many random stratified splits and reports mean and standard deviation, so
the spread is visible rather than hidden behind one lucky draw.

Protocol, identical to train_tflite.py:
  - hold out whole sessions, never rows (1 Hz samples are near duplicates)
  - stratify the hold-out across the rh_condition x label cells
  - tune every threshold on the training split only
  - fit the temperature/humidity compensation on clean-air rows only

Usage:
    python analysis/repeated_splits.py [--splits 12] [--out results/repeated_splits.json]

Author: Ayaan Khan
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["rs_ro", "temp_c", "rh_pct", "drs_dt"]


def tune_threshold(score, y):
    """Pick an alarm point by balanced accuracy.

    Searches a dense linear grid rather than quantiles. The data is strongly
    bimodal, so quantiles cluster inside the two groups and almost none land
    in the gap between them, which parks the threshold at the edge of the
    safe cluster where any small excursion trips it. Among thresholds that
    tie, this takes the midpoint of the widest tying run, which is where an
    engineer setting a trip point by hand would put it.
    """
    grid = np.linspace(float(score.min()), float(score.max()), 2000)
    scores = []
    for t in grid:
        p = (score < t).astype(int)
        tp = ((p == 1) & (y == 1)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        tn = ((p == 0) & (y == 0)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        scores.append((sens + spec) / 2)
    scores = np.asarray(scores)
    best = np.where(scores >= scores.max() - 1e-12)[0]
    runs = np.split(best, np.where(np.diff(best) != 1)[0] + 1)
    return float(grid[max(runs, key=len)].mean())


def stratified_session_split(df, rng, frac=0.3):
    """Hold out `frac` of the sessions from each rh_condition x label cell."""
    test = []
    for _, group in df.groupby(["rh_condition", "label"]):
        sessions = sorted(group.session_id.unique())
        k = max(1, int(round(len(sessions) * frac)))
        test += list(rng.choice(sessions, size=k, replace=False))
    return df[~df.session_id.isin(test)], df[df.session_id.isin(test)]


def score(y, pred, test_df):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    humid = (test_df.rh_condition == "humid").values & (y == 0)
    dry = (test_df.rh_condition == "dry").values & (y == 0)
    return {
        "accuracy": (tp + tn) / len(y),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1) if tp + fp else 1.0,
        "humid_fpr": float(pred[humid].mean()) if humid.sum() else 0.0,
        "dry_fpr": float(pred[dry].mean()) if dry.sum() else 0.0,
    }


def build_network(seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((len(FEATURES),)),
            tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sniff_sessions_run2.csv")
    ap.add_argument("--splits", type=int, default=12)
    ap.add_argument("--out", default="results/repeated_splits.json")
    args = ap.parse_args()

    import tensorflow as tf

    df = pd.read_csv(args.data)
    df["y"] = (df.label == "leak").astype(int)
    print(f"{df.session_id.nunique()} sessions, {len(df):,} samples")

    runs = {"fixed": [], "compensated": [], "network": []}

    for i in range(args.splits):
        rng = np.random.default_rng(100 + i)
        train, test = stratified_session_split(df, rng)
        y_train, y_test = train.y.values, test.y.values

        # 1. conventional fixed threshold on the raw reading
        t0 = tune_threshold(train.rs_ro.values, y_train)
        runs["fixed"].append(score(y_test, (test.rs_ro.values < t0).astype(int), test))

        # 2. threshold on the residual after temperature/humidity compensation,
        #    with the compensation fitted on clean air only
        clean = train[train.label == "safe"]
        lr = LinearRegression().fit(clean[["temp_c", "rh_pct"]].values, clean.rs_ro.values)
        res_train = train.rs_ro.values - lr.predict(train[["temp_c", "rh_pct"]].values)
        res_test = test.rs_ro.values - lr.predict(test[["temp_c", "rh_pct"]].values)
        t1 = tune_threshold(res_train, y_train)
        runs["compensated"].append(score(y_test, (res_test < t1).astype(int), test))

        # 3. the 4 -> 16 -> 8 -> 2 network
        X_train = train[FEATURES].values.astype("float32")
        X_test = test[FEATURES].values.astype("float32")
        mu, sd = X_train.mean(0), X_train.std(0)
        sd[sd < 1e-9] = 1
        Z_train, Z_test = (X_train - mu) / sd, (X_test - mu) / sd

        sessions = sorted(train.session_id.unique())
        val_sessions = set(sessions[: max(1, len(sessions) // 6)])
        val = train.session_id.isin(val_sessions).values
        if val.sum() == 0 or (~val).sum() == 0:
            val = np.zeros(len(train), bool)
            val[::6] = True

        model = build_network(i)
        model.fit(
            Z_train[~val],
            y_train[~val],
            epochs=300,
            batch_size=32,
            verbose=0,
            validation_data=(Z_train[val], y_train[val]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=25, restore_best_weights=True, monitor="val_loss"
                )
            ],
        )
        pred = np.argmax(model.predict(Z_test, verbose=0), axis=1)
        runs["network"].append(score(y_test, pred, test))
        print(f"  split {i + 1}/{args.splits}", flush=True)

    labels = {
        "fixed": "Fixed threshold",
        "compensated": "Threshold + T/RH compensation",
        "network": "TinyML 4-16-8-2",
    }
    summary = {"splits": args.splits, "protocol": "stratified session hold-out, frac=0.3"}

    print(f"\n{args.splits} random session splits")
    print("=" * 78)
    print(f"{'Detector':34s}{'Accuracy':>16s}{'Recall':>12s}{'Humid FP':>16s}")
    for key, rows in runs.items():
        acc = np.array([r["accuracy"] for r in rows])
        rec = np.array([r["recall"] for r in rows])
        hum = np.array([r["humid_fpr"] for r in rows])
        dry = np.array([r["dry_fpr"] for r in rows])
        summary[key] = {
            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
            "recall_mean": float(rec.mean()), "recall_std": float(rec.std()),
            "humid_fpr_mean": float(hum.mean()), "humid_fpr_std": float(hum.std()),
            "dry_fpr_mean": float(dry.mean()), "dry_fpr_std": float(dry.std()),
        }
        print(f"{labels[key]:34s}{acc.mean() * 100:9.1f}% ±{acc.std() * 100:4.1f}"
              f"{rec.mean() * 100:10.1f}%{hum.mean() * 100:12.1f}% ±{hum.std() * 100:4.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
