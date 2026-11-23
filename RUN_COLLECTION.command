#!/bin/bash
# SNIFF - collection runner (v3)
# Ayaan Khan
#
# DESIGN NOTES
#
# v1 failed: it checked the reader PROCESS was alive, not that BYTES arrived,
# and captured nothing for 80 minutes. v2 added a byte gate. v3 adds the
# experimental control that actually matters.
#
# The MQ-6 baseline drifts during preheat. Measured on this unit: rs_ro read
# 0.758 at 52% RH and 1.480 at 51.8% RH two and a half hours later. Same air,
# factor of two apart. If all the dry sessions run before all the humid ones,
# that drift sits exactly on top of the humidity contrast and the two cannot
# be separated. The result would be worthless and would not survive review.
#
# So the dry sessions BRACKET the humid ones:
#
#   dry (r1) -> humidify -> humid (r1,r2) -> dry out -> dry (r2)
#
# The two dry no-gas sessions are recorded at opposite ends of the run under
# the same nominal conditions. The difference between them IS the drift over
# the window, measured rather than assumed, and this script reports it. If it
# is small relative to the humidity effect, the comparison stands. If it is
# large, you know that before you write the report instead of after.
#
# Gas and no-gas alternate within every round, so the leak/safe contrast is
# never aligned with elapsed time either.

cd "$(dirname "$0")" || exit 1

DUR=300
SETTLE=20

RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; BLD=$'\033[1m'; OFF=$'\033[0m'

echo ""
echo "${BLD}SNIFF data collection (v3)${OFF}"
echo "Eight sessions, ${DUR}s each. Dry sessions bracket the humid round."
echo ""

PORT=$(ls /dev/cu.usbserial-* /dev/cu.wchusbserial* /dev/cu.SLAB_USBtoUART* 2>/dev/null | head -1)
[ -z "$PORT" ] && { echo "${RED}No ESP32 found.${OFF}"; echo "Press any key."; read -r -n 1; exit 1; }
echo "Port: ${GRN}${PORT}${OFF}"

if lsof "$PORT" >/dev/null 2>&1; then
  echo "${RED}Port busy.${OFF} Quit Arduino IDE entirely (Cmd+Q), then rerun."
  echo "Press any key."; read -r -n 1; exit 1
fi

mkdir -p data raw
STAMP=$(date +%Y%m%d_%H%M%S)
RAW="raw/stream_${STAMP}.log"
MARKS="raw/marks_${STAMP}.csv"
: > "$MARKS"; : > "$RAW"

exec 3< "$PORT" || { echo "${RED}Cannot open port.${OFF}"; exit 1; }
stty -f "$PORT" 115200 raw -echo -hupcl 2>/dev/null

perl -e '$| = 1; while (my $l = <STDIN>) { $l =~ s/\r//g; print time(), ",", $l; }' <&3 >> "$RAW" &
READER=$!
cleanup() { kill "$READER" 2>/dev/null; exec 3<&- 2>/dev/null; }
trap cleanup EXIT

# Counts valid data rows. Strips NULs and CRs first: the serial stream carries
# NUL bytes at connect, which makes BSD grep treat the file as binary and
# report zero matches even when every line is good. That false negative is
# what made the v2 port check fail on perfectly healthy data.
count_rows () {
  tr -d '\000\r' < "$1" 2>/dev/null | grep -cE '^[0-9]+,[0-9.+-]+,[0-9.+-]+,' 2>/dev/null || true
}

echo ""
echo "${BLD}Verifying the board is actually sending data...${OFF}"
GOOD=0
for attempt in 1 2 3; do
  for i in $(seq 1 12); do
    ROWS=$(count_rows "$RAW"); ROWS=${ROWS:-0}
    printf "\r  attempt %d   %3d valid rows  " "$attempt" "$ROWS"
    [ "$ROWS" -ge 5 ] && { GOOD=1; break; }
    sleep 1
  done
  echo ""
  [ "$GOOD" = "1" ] && break
  echo "${YEL}  Nothing yet. Press EN (or RST) on the ESP32 now.${OFF}"
  sleep 3
done

if [ "$GOOD" != "1" ]; then
  echo ""
  echo "${RED}${BLD}STOPPING.${OFF} The board is not sending anything."
  echo "Nothing recorded, nothing lost. Bytes seen: $(wc -c < "$RAW" | tr -d ' ')"
  echo ""; echo "Press any key."; read -r -n 1; exit 1
fi
echo "${GRN}  Data confirmed.${OFF}"
echo ""

# NOTE: the firmware on the board here is sniff_web, whose serial line is
#   rs_ro,temp_c,rh_pct,drs_dt,model,confidence,threshold
# The logger's line is
#   millis,aout_v,rs_kohm,rs_ro,temp_c,rh_pct,drs_dt,baseline_alarm
# The splitter below accepts the 8-column logger format only. Flash
# sniff_logger before running this.

live_line () {
  local L rs t rh a
  L=$(tail -n 1 "$RAW" 2>/dev/null)
  IFS=, read -r _ _ _ _ rs t rh _ a <<< "$L"
  if [ -n "$rh" ]; then
    local flag="safe"; [ "$a" = "1" ] && flag="ALARM"
    printf "Rs/Ro %-7s  T %-5s C  RH %-5s %%  [%s]" "$rs" "$t" "$rh" "$flag"
  else
    printf "waiting for data"
  fi
}

live_until_enter () {
  ( while true; do printf "\r  %-62s" "$(live_line)"; sleep 3; done ) &
  local W=$!
  read -r
  kill "$W" 2>/dev/null; wait "$W" 2>/dev/null
  echo ""
}

session () {
  local gas="$1" rh="$2" rep="$3" msg="$4"
  local label="safe"; [ "$gas" -gt 0 ] && label="leak"
  local sid="S_g${gas}_${rh}_r${rep}"

  echo "${BLD}──────────────────────────────────────────────${OFF}"
  echo "${BLD}${sid}${OFF}   gas=${gas} ppm   rh=${rh}   rep ${rep}"
  echo ""
  echo "${YEL}${msg}${OFF}"
  echo ""
  printf "Press ENTER when ready.\n"
  live_until_enter

  local before=$(count_rows "$RAW"); before=${before:-0}
  echo -n "  settling ${SETTLE}s"; sleep "$SETTLE"; echo ""
  local t0=$(date +%s) i=0
  while [ "$i" -lt "$DUR" ]; do
    sleep 5; i=$((i+5))
    printf "\r  %3ds/%ds  %-58s" "$i" "$DUR" "$(live_line)"
  done
  local t1=$(date +%s)
  local after=$(count_rows "$RAW"); after=${after:-0}
  echo ""
  local got=$((after - before))
  if [ "$got" -lt 60 ]; then
    echo "${RED}  WARNING: only $got rows this session. Board may have dropped out.${OFF}"
  else
    echo "${GRN}  done, ~$got rows${OFF}"
  fi
  echo ""
  echo "${t0},${t1},${sid},${gas},${rh},${label},rep ${rep}" >> "$MARKS"
}

echo "${BLD}DRY, FIRST HALF OF THE BRACKET${OFF}"
echo ""
session 0 dry 1 "Box empty except the board. No cloth. Lid on. No gas.
  This is the opening baseline."
session 2000 dry 1 "Lid off. Can UPRIGHT, six inches from the sensor, aimed at a
  far wall. HALF A SECOND. Lid on. Rock twice."

echo "${BLD}──────────────────────────────────────────────${OFF}"
echo "${BLD}HUMIDIFY${OFF}"
echo ""
echo "  Lid off by an open window for five minutes to clear the gas."
echo "  Then put the damp cloth in, not touching the sensor. Lid on."
echo "  Wait for RH to stop climbing. Watch the number below, not the clock."
echo ""
printf "Press ENTER once RH has levelled off.\n"
live_until_enter

echo "${BLD}HUMID ROUND${OFF}"
echo ""
session 0 humid 1 "Cloth in, lid on, NO GAS.
  ${BLD}This is the session the whole project rests on.${OFF}"
session 2000 humid 1 "Lid off, upright can, half-second burst, lid on, rock twice."

echo "  ${YEL}VENTILATE: lid off by the window, five minutes. Cloth stays in.${OFF}"
echo "  ${YEL}Re-wet and wring it hard if it has dried out.${OFF}"
echo ""
printf "Press ENTER once RH is back up and the gas is cleared.\n"
live_until_enter

session 0 humid 2 "Cloth in, lid on, no gas. Second replicate."
session 2000 humid 2 "Lid off, upright can, half-second burst, lid on, rock twice."

echo "${BLD}──────────────────────────────────────────────${OFF}"
echo "${BLD}DRY THE BOX OUT - SECOND HALF OF THE BRACKET${OFF}"
echo ""
echo "  Take the cloth OUT. Wipe condensation off the walls."
echo "  Lid off by the window, ten minutes."
echo "  Wait for RH to come back to roughly where it was at the very start."
echo "  Getting this close to the opening RH is what makes the drift"
echo "  measurement valid, so do not rush it."
echo ""
printf "Press ENTER once RH is back near its opening value.\n"
live_until_enter

session 0 dry 2 "Box empty, lid on, no gas.
  ${BLD}This closes the bracket. Same conditions as session 1.${OFF}"
session 2000 dry 2 "Last one. Lid off, upright can, half-second burst, lid on, rock twice."

kill "$READER" 2>/dev/null; exec 3<&- 2>/dev/null
sleep 1
echo "${BLD}──────────────────────────────────────────────${OFF}"
echo "Splitting the stream into labelled sessions..."

perl -e '
  my ($raw, $marks) = @ARGV;
  my @m;
  open(my $M, "<", $marks) or die $!;
  while (<$M>) { chomp; my @f = split /,/, $_, 7; push @m, \@f if @f == 7; }
  close $M;

  my %rows;
  open(my $R, "<", $raw) or die $!;
  while (my $l = <$R>) {
    chomp $l;
    my ($ts, $rest) = split /,/, $l, 2;
    next unless defined $rest;
    next if $rest =~ /^\s*#/;
    next unless $rest =~ /^\d+,/;
    my @c = split /,/, $rest;
    next unless @c == 8;
    for my $mk (@m) {
      if ($ts >= $mk->[0] && $ts <= $mk->[1]) { push @{ $rows{$mk->[2]} }, [$ts, @c]; last; }
    }
  }
  close $R;

  my (%meanRsRo, %meanRH, $total);
  for my $mk (@m) {
    my ($t0,$t1,$sid,$gas,$rh,$label,$notes) = @$mk;
    my $r = $rows{$sid} || [];
    open(my $O, ">", "data/$sid.csv") or die $!;
    print $O "session_id,timestamp,millis,aout_v,rs_kohm,rs_ro,temp_c,rh_pct,drs_dt,baseline_alarm,gas_ppm_nominal,rh_condition,label,notes\n";
    my ($sr, $sh, $n) = (0,0,0);
    for my $row (@$r) {
      my ($ts, @c) = @$row;
      my @lt = localtime($ts);
      my $iso = sprintf("%04d-%02d-%02dT%02d:%02d:%02d", $lt[5]+1900, $lt[4]+1, $lt[3], $lt[2], $lt[1], $lt[0]);
      print $O join(",", $sid, $iso, @c, $gas, $rh, $label, "recorded - $notes"), "\n";
      $sr += $c[3]; $sh += $c[5]; $n++;
    }
    close $O;
    if ($n) { $meanRsRo{$sid} = $sr/$n; $meanRH{$sid} = $sh/$n; }
    printf("  %-24s %5d rows\n", "$sid.csv", scalar @$r);
    $total += scalar @$r;
  }
  print "\n  $total rows total across ", scalar(@m), " sessions\n";
  if ($total == 0) { print "\n  *** ZERO ROWS - stop and diagnose, do not rerun ***\n"; exit; }

  my ($a, $b) = ("S_g0_dry_r1", "S_g0_dry_r2");
  if (exists $meanRsRo{$a} && exists $meanRsRo{$b}) {
    my $drift = $meanRsRo{$b} - $meanRsRo{$a};
    my $pct   = 100 * $drift / $meanRsRo{$a};
    printf("\n  DRIFT BRACKET (dry, no gas, start vs end)\n");
    printf("    start  rs_ro %.4f  at RH %.1f%%\n", $meanRsRo{$a}, $meanRH{$a});
    printf("    end    rs_ro %.4f  at RH %.1f%%\n", $meanRsRo{$b}, $meanRH{$b});
    printf("    drift  %+.4f  (%+.1f%%) over the run\n", $drift, $pct);
    my $h = $meanRsRo{"S_g0_humid_r1"};
    if (defined $h) {
      my $heff = $h - $meanRsRo{$a};
      printf("    humidity effect (dry start -> humid)  %+.4f  (%+.1f%%)\n",
             $heff, 100*$heff/$meanRsRo{$a});
      printf("    -> drift is %.0f%% the size of the humidity effect\n",
             100*abs($drift)/(abs($heff) || 1e-9));
    }
  }
' "$RAW" "$MARKS"

echo ""
echo "${GRN}${BLD}DONE.${OFF} Raw stream at ${RAW}"
echo ""
echo "Press any key to close."
read -r -n 1
