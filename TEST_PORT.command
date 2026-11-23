#!/bin/bash
# SNIFF - 15 second port check. Run this BEFORE collecting.
cd "$(dirname "$0")" || exit 1
RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; BLD=$'\033[1m'; OFF=$'\033[0m'

echo ""; echo "${BLD}SNIFF port check${OFF}"; echo ""

PORT=$(ls /dev/cu.usbserial-* /dev/cu.wchusbserial* /dev/cu.SLAB_USBtoUART* 2>/dev/null | head -1)
[ -z "$PORT" ] && { echo "${RED}No ESP32 found.${OFF}"; echo "Press any key."; read -r -n 1; exit 1; }
echo "Port: $PORT"

if lsof "$PORT" >/dev/null 2>&1; then
  echo "${RED}Port busy.${OFF} Quit Arduino IDE entirely (Cmd+Q), then rerun."
  echo "Press any key."; read -r -n 1; exit 1
fi

# Counts valid data rows. Strips NULs and CRs first: the serial stream carries
# NUL bytes at connect, which makes BSD grep treat the file as binary and
# report zero matches even when every line is good. That false negative is
# what made the v2 port check fail on perfectly healthy data.
count_rows () {
  tr -d '\000\r' < "$1" 2>/dev/null | grep -cE '^[0-9]+,[0-9.+-]+,[0-9.+-]+,' 2>/dev/null || true
}

OUT=$(mktemp)
# Open the fd FIRST, then apply termios to the already-open port.
# stty on a closed cu device gets reverted when the next process opens it.
exec 3< "$PORT" || { echo "${RED}Cannot open port.${OFF}"; exit 1; }
stty -f "$PORT" 115200 raw -echo -hupcl 2>/dev/null
cat <&3 > "$OUT" &
CATPID=$!

echo ""
echo "Listening 15 s. If nothing arrives, ${BLD}press the EN (or RST) button${OFF}"
echo "on the ESP32 board now - the port open may have left it in bootloader."
echo ""
for i in $(seq 1 15); do
  B=$(wc -c < "$OUT" | tr -d ' ')
  printf "\r  %2ds   %6s bytes  " "$i" "$B"
  sleep 1
done
echo ""
kill "$CATPID" 2>/dev/null; exec 3<&-

BYTES=$(wc -c < "$OUT" | tr -d ' ')
ROWS=$(count_rows "$OUT"); ROWS=${ROWS:-0}
echo ""
echo "  bytes: $BYTES"
echo "  valid CSV rows: $ROWS"
echo ""
echo "${BLD}Sample of what came back:${OFF}"
head -c 400 "$OUT" | tr -d '\000'
echo ""; echo ""

if [ "$ROWS" -ge 5 ]; then
  echo "${GRN}${BLD}PASS.${OFF} Port is good. Run RUN_COLLECTION.command now."
elif [ "$BYTES" -gt 0 ]; then
  echo "${YEL}${BLD}GARBAGE.${OFF} Bytes arrived but not valid rows - baud mismatch."
  echo "Check that the sample above looks like valid CSV rows."
else
  echo "${RED}${BLD}FAIL.${OFF} Nothing at all. Press EN on the board and run this again."
fi
rm -f "$OUT"
echo ""; echo "Press any key to close."; read -r -n 1
