#!/usr/bin/env python3
"""
SNIFF session logger.
Author: Ayaan Khan

Reads the CSV stream from the ESP32 and writes it with session metadata
attached. Labels come from the protocol, not the sensor: you control the
exposure, so you know the ground truth.

Session metadata matters more than it looks. Training holds out whole
sessions rather than random rows, and that is only possible if every row
knows which session it came from.

Usage
-----
    python collect.py --list-ports

    python collect.py --port /dev/cu.usbserial-0001 \
        --gas 0 --rh dry --duration 900 --notes "room air, sensor A"

    python collect.py --port COM3 --gas 2000 --rh humid --duration 900

Install
-------
    pip install pyserial
"""

import argparse
import csv
import datetime as dt
import os
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial missing. Run:  pip install pyserial")

FIELDS = [
    "session_id", "timestamp", "millis", "aout_v", "rs_kohm", "rs_ro",
    "temp_c", "rh_pct", "drs_dt", "baseline_alarm",
    "gas_ppm_nominal", "rh_condition", "label", "notes",
]


def show_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the ESP32 plugged in with a data cable?")
        return
    print("Serial ports:")
    for p in ports:
        print(f"  {p.device:28s} {p.description}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    ap.add_argument("--port", help="serial port of the ESP32")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--gas", type=float, help="nominal injected gas in ppm; 0 for clean air")
    ap.add_argument("--rh", choices=["dry", "ambient", "humid"],
                    help="humidity condition for this session")
    ap.add_argument("--duration", type=int, default=900, help="seconds to record")
    ap.add_argument("--notes", default="")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    if args.list_ports:
        show_ports()
        return

    missing = [n for n, v in [("--port", args.port), ("--gas", args.gas), ("--rh", args.rh)]
               if v is None]
    if missing:
        sys.exit(f"Missing required argument(s): {', '.join(missing)}\n"
                 f"Run with --list-ports to find your port.")

    if args.gas > 2000:
        sys.exit(f"Refusing: {args.gas:.0f} ppm is above the 2000 ppm safety cap "
                 f"(10% of propane's lower explosive limit).")

    # Ground truth comes from the protocol. We injected it, so we know.
    label = "leak" if args.gas > 0 else "safe"

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"g{int(args.gas)}_{args.rh}_{stamp}"

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{session_id}.csv")

    print(f"session  : {session_id}")
    print(f"label    : {label}")
    print(f"condition: {args.rh} air, {args.gas:.0f} ppm nominal")
    print(f"writing  : {path}")
    print(f"duration : {args.duration} s ({args.duration/60:.0f} min)\n")

    if args.gas > 0:
        print("Before you inject: ventilation on, no ignition sources, nothing hot on the bench.\n")

    rows = bad = 0
    try:
        with serial.Serial(args.port, args.baud, timeout=2) as ser, \
             open(path, "w", newline="") as fh:

            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            start = time.time()

            while time.time() - start < args.duration:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                if raw.startswith("#"):
                    print(f"  [esp32] {raw}")
                    continue

                parts = raw.split(",")
                if len(parts) != 8:
                    bad += 1
                    continue

                try:
                    writer.writerow({
                        "session_id":      session_id,
                        "timestamp":       dt.datetime.now().isoformat(timespec="seconds"),
                        "millis":          int(parts[0]),
                        "aout_v":          float(parts[1]),
                        "rs_kohm":         float(parts[2]),
                        "rs_ro":           float(parts[3]),
                        "temp_c":          float(parts[4]),
                        "rh_pct":          float(parts[5]),
                        "drs_dt":          float(parts[6]),
                        "baseline_alarm":  int(parts[7]),
                        "gas_ppm_nominal": args.gas,
                        "rh_condition":    args.rh,
                        "label":           label,
                        "notes":           args.notes,
                    })
                except ValueError:
                    bad += 1
                    continue

                rows += 1
                fh.flush()
                if rows % 30 == 0:
                    el = int(time.time() - start)
                    print(f"  {rows:5d} rows | {el:4d}/{args.duration} s | "
                          f"Rs/Ro {parts[3]:>7s} | RH {parts[5]:>5s}% | T {parts[4]:>5s}C")

    except KeyboardInterrupt:
        print("\nStopped early. Partial session saved.")
    except serial.SerialException as e:
        sys.exit(f"\nSerial error: {e}\nCheck the port and that no other program "
                 f"(Arduino Serial Monitor) has it open.")

    print(f"\ndone. {rows} rows written, {bad} malformed lines skipped.")
    if rows == 0:
        print("WARNING: nothing recorded. Is the firmware printing CSV at 115200?")
    elif bad > rows * 0.05:
        print("WARNING: high malformed-line rate. Check baud rate and cable.")


if __name__ == "__main__":
    main()
