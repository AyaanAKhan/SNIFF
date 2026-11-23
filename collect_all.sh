#!/usr/bin/env bash
# SNIFF - guided data collection, then retrain
# Ayaan Khan
#
#   bash ~/Documents/SNIFF/collect_all.sh
#
# Records 8 sessions (2 replicates x 4 conditions), ~25 minutes total,
# then clears the simulated data and retrains on what you recorded.
# Two replicates matter: training holds out whole sessions, and with only
# one per condition there is nothing to hold out.

set -u
cd "$HOME/Documents/SNIFF" || { echo "No ~/Documents/SNIFF"; exit 1; }

DUR=180            # seconds per session
PORT=""

hr()  { printf "\n\033[1;34m%s\033[0m\n" "------------------------------------------------------------"; }
say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
ask() { printf "\n\033[1;33m%s\033[0m\n    Press Return when ready. " "$*"; read -r _; }

# ---------------------------------------------------------------- deps
python3 -c "import serial" 2>/dev/null || { say "Installing pyserial"; pip3 install --quiet pyserial; }

# ---------------------------------------------------------------- port
say "Finding the ESP32"
PORT=$(python3 - <<'PY'
from serial.tools import list_ports
for p in list_ports.comports():
    d = p.device
    if any(k in d for k in ("usbserial","usbmodem","wchusbserial","SLAB_USBtoUART")):
        print(d); break
PY
)
[ -n "$PORT" ] || { echo "No board found. Plug it in and rerun."; exit 1; }
say "Using $PORT"

echo
echo "IMPORTANT: close the Arduino Serial Monitor before continuing."
echo "Only one program can hold the serial port at a time."
ask "Serial Monitor closed?"

# ---------------------------------------------------------------- flash the logger?
cat <<'NOTE'

NOTE: your board is currently running the INFERENCE firmware, which prints
a different set of columns than the logger. collect.py expects the logger.

If the first session errors with "malformed lines", stop, flash
firmware/sniff_logger in Arduino IDE, and rerun this script.

NOTE
ask "Continue?"

run() {   # run <gas> <rh> <rep> <instruction>
  local gas="$1" rh="$2" rep="$3" msg="$4"
  hr
  say "Session: $rh air, $( [ "$gas" = 0 ] && echo 'no gas' || echo 'alcohol' ), replicate $rep"
  ask "$msg"
  python3 collect.py --port "$PORT" --gas "$gas" --rh "$rh" \
      --duration "$DUR" --notes "rep $rep, ethanol proxy" || return 1
}

for rep in 1 2; do
  run 0 dry "$rep" \
    "DRY, NO GAS.
    Sensor in open room air. Nothing near it. Do not breathe on it."

  run 1500 dry "$rep" \
    "DRY + ALCOHOL.
    Put a cotton ball or tissue with hand sanitizer in the container
    with the sensor, lid on. Keep it a few cm away, not touching."

  hr
  ask "VENTILATE. Lid off, let it clear for 2 minutes, then continue."

  run 0 humid "$rep" \
    "HUMID, NO GAS.  <-- the most important session
    Damp paper towel in the container, lid on, no alcohol.
    Wait until the humidity reading stops climbing, then continue."

  run 1500 humid "$rep" \
    "HUMID + ALCOHOL.
    Keep the damp towel in, add the sanitizer cotton ball, lid on."

  hr
  [ "$rep" = 1 ] && ask "Replicate 1 done. Ventilate, dry the container, then continue."
done

# ---------------------------------------------------------------- retrain
hr
say "Clearing simulated sessions"
python3 make_synthetic.py --clean

say "Training on your recorded data"
python3 train.py || { echo "Training failed. See the error above."; exit 1; }

hr
printf "\033[1;32mDone.\033[0m model.h is rebuilt from measured data.\n"
printf "Reflash the board, or upload it yourself in Arduino IDE.\n\n"
