#!/usr/bin/env bash
# SNIFF - short guided data collection, then retrain
# Ayaan Khan
#
#   bash ~/Documents/SNIFF/collect_quick.sh
#
# 8 sessions x 2 minutes = ~20 minutes with transitions.
# Two replicates per condition, because training holds out whole sessions
# and with one session per condition there is nothing to hold out.

set -u
cd "$HOME/Documents/SNIFF" || { echo "No ~/Documents/SNIFF"; exit 1; }
DUR=120

hr()  { printf "\n\033[1;34m%s\033[0m\n" "============================================================"; }
say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
ask() { printf "\n\033[1;33m%s\033[0m\n\n    Press Return when ready. " "$*"; read -r _; }

python3 -c "import serial" 2>/dev/null || { say "Installing pyserial"; pip3 install --quiet pyserial; }

say "Finding the board"
PORT=$(python3 - <<'PY'
from serial.tools import list_ports
for p in list_ports.comports():
    if any(k in p.device for k in ("usbserial","usbmodem","wchusbserial","SLAB_USBtoUART")):
        print(p.device); break
PY
)
[ -n "$PORT" ] || { echo "No board found. Plug it in and rerun."; exit 1; }
say "Using $PORT"

hr
cat <<'INTRO'
WHAT YOU NEED

  A container        any tupperware, a jar, or a ziploc bag
  A smell source     cologne, deodorant, hairspray, nail polish remover,
                     a Sharpie, or any cleaning spray
  Water              for a damp paper towel

The smell source stands in for propane. It is a reducing gas the MQ-6
responds to through the same mechanism, which is the point being tested.
Your write-up calls it what it is: a VOC proxy, not calibrated to ppm.

Close the Arduino Serial Monitor before continuing. Only one program can
hold the serial port.
INTRO
ask "Ready?"

run() {   # run <gas> <rh> <rep> <instruction>
  hr
  say "$2 air, $( [ "$1" = 0 ] && echo 'NO source' || echo 'WITH source' ), replicate $3"
  ask "$4"
  python3 collect.py --port "$PORT" --gas "$1" --rh "$2" --duration "$DUR" \
      --notes "rep $3, VOC proxy, uncalibrated concentration" || return 1
}

for rep in 1 2; do
  run 0 ambient "$rep" \
"AMBIENT, NOTHING NEAR IT.
    Sensor in open room air. Do not breathe on it."

  run 1000 ambient "$rep" \
"AMBIENT + SOURCE.
    Put the sensor and the smell source in the container together.
    Keep them a few cm apart. Lid on. Nothing touching the mesh."

  hr
  ask "VENTILATE. Lid off, 2 minutes, let it clear completely."

  run 0 humid "$rep" \
"HUMID, NO SOURCE.   <-- the session your whole project rests on
    Damp paper towel in the container with the sensor. No smell source.
    Lid on. Give it a minute for the humidity to come up."

  run 1000 humid "$rep" \
"HUMID + SOURCE.
    Keep the damp towel in, add the smell source back. Lid on."

  hr
  [ "$rep" = 1 ] && ask "Round 1 done. Ventilate, dry the container, then go again."
done

hr
say "Clearing the simulated sessions"
python3 make_synthetic.py --clean
say "Training on your recorded data"
python3 train.py || { echo "Training failed. See the error above."; exit 1; }

hr
printf "\033[1;32mDone.\033[0m model.h is rebuilt from data you measured.\n"
printf "Reflash the board.\n\n"
