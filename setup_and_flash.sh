#!/usr/bin/env bash
# SNIFF - one-shot setup and flash
# Ayaan Khan
#
# Installs the toolchain, compiles the inference firmware and uploads it
# to the ESP32. Run once; after that, reflashing is just the last step.
#
#   bash ~/Documents/SNIFF/setup_and_flash.sh
#
set -u

SKETCH="$HOME/Documents/SNIFF/firmware/sniff_inference"
FQBN="esp32:esp32:esp32"
BOARD_URL="https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json"

say()  { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }
ok()   { printf "    \033[32mok\033[0m  %s\n" "$*"; }
die()  { printf "\n\033[31mSTOP:\033[0m %s\n\n" "$*"; exit 1; }

# ---------------------------------------------------------------- checks
say "Checking the sketch"
[ -d "$SKETCH" ] || die "No sketch folder at $SKETCH"
[ -f "$SKETCH/model.h" ] || die "model.h is missing. Run: cd ~/Documents/SNIFF && python3 train.py"
ok "sketch and model.h present"

RL=$(grep -m1 'RL_KOHM *=' "$SKETCH/sniff_inference.ino" | sed 's/[^0-9.]*\([0-9.]*\).*/\1/')
printf "\n    RL_KOHM is currently %s kohm.\n" "$RL"
printf "    If you have not measured your MQ-6 load resistor, stop and do that\n"
printf "    first, or every resistance reading will be wrong.\n"
printf "\n    Press Return to continue, or Ctrl-C to go measure it. "
read -r _

# ---------------------------------------------------------------- toolchain
say "Checking for arduino-cli"
if ! command -v arduino-cli >/dev/null 2>&1; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

if ! command -v arduino-cli >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing arduino-cli with Homebrew (a few minutes)"
    brew install arduino-cli || die "brew install failed"
    export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
  else
    die "Neither arduino-cli nor Homebrew is installed.

Pick one:
  A) Install Homebrew, then rerun this script:
     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"

  B) Install Arduino IDE by hand from https://arduino.cc/en/software
     and follow the steps in RUN_TODAY.md instead."
  fi
fi
ok "arduino-cli $(arduino-cli version | head -1)"

# ---------------------------------------------------------------- esp32 core
say "Configuring the ESP32 board package"
arduino-cli config init --overwrite >/dev/null 2>&1 || true
arduino-cli config set board_manager.additional_urls "$BOARD_URL" >/dev/null 2>&1 \
  || arduino-cli config add board_manager.additional_urls "$BOARD_URL" >/dev/null 2>&1
arduino-cli core update-index || die "Could not reach the board index. Check your network."

if arduino-cli core list 2>/dev/null | grep -q '^esp32:esp32'; then
  ok "esp32 core already installed"
else
  say "Installing the ESP32 core (large download, be patient)"
  arduino-cli core install esp32:esp32 || die "ESP32 core install failed"
  ok "esp32 core installed"
fi

# ---------------------------------------------------------------- library
say "Installing the DHT library"
arduino-cli lib install "DHT sensor library" >/dev/null 2>&1 \
  && ok "DHT sensor library" || ok "DHT sensor library (already present)"
arduino-cli lib install "Adafruit Unified Sensor" >/dev/null 2>&1 \
  && ok "Adafruit Unified Sensor" || ok "Adafruit Unified Sensor (already present)"

# ---------------------------------------------------------------- compile
say "Compiling"
arduino-cli compile --fqbn "$FQBN" "$SKETCH" || die "Compile failed. Read the error above."
ok "compiled clean"

# ---------------------------------------------------------------- port
say "Looking for the board"
arduino-cli board list
PORT=$(arduino-cli board list 2>/dev/null \
       | awk '/usbserial|usbmodem|wchusbserial|SLAB_USBtoUART/ {print $1; exit}')

if [ -z "${PORT:-}" ]; then
  die "No board found.

  - Is the ESP32 plugged in?
  - Is the cable a DATA cable? Charge-only cables will not work.
  - You may need the USB driver: CP2102 (Silicon Labs) or CH340.

  Firmware compiled fine, so once the board shows up just rerun this."
fi
ok "found $PORT"

# ---------------------------------------------------------------- upload
say "Uploading to $PORT"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH" || die "Upload failed.

  Try holding the BOOT button on the ESP32 while the upload starts."

printf "\n\033[1;32m================ DONE ================\033[0m\n"
printf "Firmware is on the board.\n\n"
printf "Watch it run:\n"
printf "  arduino-cli monitor -p %s -c baud=115200\n\n" "$PORT"
printf "Columns: rs_ro,temp_c,rh_pct,drs_dt,model,confidence,threshold_says\n"
printf "Send 'c' to calibrate Ro in clean air. Send 'm' to mute the buzzer.\n\n"
