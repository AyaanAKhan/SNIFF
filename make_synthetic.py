#!/usr/bin/env python3
"""
SNIFF - simulated session generator.
Author: Ayaan Khan

Generates physically-grounded fake sensor sessions so the full pipeline
(train -> export -> flash -> demo) can be exercised without waiting on
chamber time.

Provenance
----------
These sessions are simulated from the physical model below, not recorded
from a sensor. They are marked so the pipeline can distinguish them:

  * session_id starts with SYNTH_
  * the notes column reads "simulated"
  * train.py reports data_source: simulated
  * model.h and results/metrics.json carry the same field

Use them to exercise the full pipeline and to drive the demo. When you
report numbers, say which dataset they came from. Mixing simulated and
recorded sessions in one training run makes the metrics describe neither,
so clear one out first with --clean.

Usage
-----
    python make_synthetic.py                 # 8 sessions, 2 reps per condition
    python make_synthetic.py --reps 3
    python make_synthetic.py --clean         # delete synthetic sessions
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

OUTDIR = "data"
MARK = "simulated - make_synthetic.py"


def session(gas_ppm, rh_condition, rh_target, temp_base, rep, seconds, rng):
    """Model the MQ-6 the way the datasheet describes it.

    Rs falls when a reducing gas reaches the tin dioxide surface, and falls
    again with water vapour through the same mechanism. Temperature pushes
    the same direction. The two differ in their time constants, which is
    the signal the classifier is meant to find:
      * a leak ramps over seconds
      * humidity drifts over minutes to hours
    """
    n = seconds
    t = np.arange(n, dtype=float)

    # Humidity: slow drift around the target, plus sensor noise.
    rh = rh_target + 3.0 * np.sin(2 * np.pi * t / 900.0) + rng.normal(0, 0.8, n)
    rh = np.clip(rh, 5, 98)

    # Temperature: slow drift.
    temp = temp_base + 0.6 * np.sin(2 * np.pi * t / 1200.0) + rng.normal(0, 0.12, n)

    # Clean-air Rs/Ro: 1.0 at the calibration point, pushed down by humidity
    # and temperature. Coefficients are in the range the datasheet's
    # temperature/humidity curve implies.
    rs_ro = 1.0 - 0.0045 * (rh - 33.0) - 0.004 * (temp - 20.0)

    if gas_ppm > 0:
        # Gas enters around 60 s and saturates over roughly 90 s.
        onset, tau = 60.0, 90.0
        ramp = np.clip((t - onset) / tau, 0.0, 1.0)
        # Response is sublinear in concentration, as the log-log
        # sensitivity curve implies.
        depth = 0.55 * (gas_ppm / 2000.0) ** 0.6
        rs_ro = rs_ro - depth * ramp
        # Turbulence while the gas disperses.
        rs_ro += rng.normal(0, 0.012, n) * ramp

    rs_ro += rng.normal(0, 0.010, n)          # baseline sensor noise
    rs_ro = np.clip(rs_ro, 0.04, 2.0)

    Ro = 9.2
    rs_kohm = rs_ro * Ro

    # dRs/dt over a 10 s window, matching the firmware.
    drs = np.zeros(n)
    drs[10:] = (rs_kohm[10:] - rs_kohm[:-10]) / 9.0

    vc, rl = 5.0, 10.0
    aout = vc * rl / (rl + rs_kohm)

    label = "leak" if gas_ppm > 0 else "safe"
    sid = f"SYNTH_g{int(gas_ppm)}_{rh_condition}_r{rep}"

    return pd.DataFrame(dict(
        session_id=sid,
        timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
        millis=(t * 1000).astype(int),
        aout_v=np.round(aout, 4),
        rs_kohm=np.round(rs_kohm, 3),
        rs_ro=np.round(rs_ro, 4),
        temp_c=np.round(temp, 2),
        rh_pct=np.round(rh, 2),
        drs_dt=np.round(drs, 5),
        baseline_alarm=(rs_ro < 0.45).astype(int),
        gas_ppm_nominal=gas_ppm,
        rh_condition=rh_condition,
        label=label,
        notes=MARK,
    ))


def clean():
    files = glob.glob(os.path.join(OUTDIR, "SYNTH_*.csv"))
    for f in files:
        os.remove(f)
    print(f"Removed {len(files)} synthetic session file(s).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2,
                    help="replicates per condition (2 minimum for a session split)")
    ap.add_argument("--seconds", type=int, default=600, help="length of each session")
    ap.add_argument("--clean", action="store_true", help="delete synthetic sessions and exit")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.clean:
        clean()
        return

    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    conditions = [
        (0,    "dry",   30.0, 21.5),
        (2000, "dry",   30.0, 21.5),
        (0,    "humid", 76.0, 22.5),   # the critical false-alarm case
        (2000, "humid", 76.0, 22.5),
    ]

    print("Generating simulated sessions. Files are prefixed SYNTH_ so the")
    print("training script can tell them apart from recorded ones.\n")

    total = 0
    for gas, cond, rh, temp in conditions:
        for rep in range(1, args.reps + 1):
            df = session(gas, cond, rh, temp, rep, args.seconds, rng)
            path = os.path.join(OUTDIR, f"{df.session_id.iloc[0]}.csv")
            df.to_csv(path, index=False)
            total += len(df)
            print(f"  {path}  {len(df)} rows  "
                  f"mean Rs/Ro {df.rs_ro.mean():.3f}  RH {df.rh_pct.mean():.0f}%")

    print(f"\n{len(conditions) * args.reps} sessions, {total} rows.")
    print("\nNext:  python train.py")
    print("When you have real data, run:  python make_synthetic.py --clean")


if __name__ == "__main__":
    main()
