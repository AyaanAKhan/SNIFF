# SNIFF — collection and demo runbook

Ayaan Khan

The hardware is assembled, the firmware is written and flashed, and the
full pipeline has been run end to end on simulated data. The demo works
right now.

What is left is recording real sessions so the model is trained on
measurements instead of a simulation. That is about five hours, and most of
it is the sensor sitting in a box recording.

**Start the clock when you get back with butane.** Nothing else is blocking.

---

## Current state of the unit

| | |
|---|---|
| Flashed sketch | `firmware/sniff_web/sniff_web.ino` |
| Ro, measured in clean air | **19.42 kΩ**, stored in flash |
| Dashboard | `SNIFF-Detector` / `sniffgas2026` → `http://192.168.4.1` |
| Model | int8 TF Lite, 3,440 bytes, trained on **simulated** data |

Ro is saved in NVS and survives a reflash. You only redo it if the sensor
is replaced or the load resistor changes.

---

## What to buy first (30 minutes)

| Item | Where | Cost |
|---|---|---|
| Butane lighter refill can | Gas station, 7-Eleven, Walmart, hardware store, smoke shop | ~$5 |
| Clear plastic container with a sealing lid, 1–2 gallon | Walmart, Target, grocery store | ~$8 |
| Oral medicine syringe, 10 mL | Any pharmacy counter, often free if you ask | $0–2 |

Ask for "lighter refill fluid" or show them a lighter. Ronson, Zippo and
Clipper all work. Any of them is butane.

The syringe is the difference between a number you can defend and a guess.
Get one if the pharmacy is on the way. If not, there's a fallback below.

You do not need salt solutions. A wet paper towel gets you to 70–90% RH,
and the DHT22 measures the actual value, so you report what you measured
rather than a nominal figure. That is honest and it is good enough for a
two-level experiment.

---

## Safety, the short version

- **Cap every exposure at 2,000 ppm.** That is a tenth of propane's lower
  explosive limit, the standard industrial margin. `collect.py` refuses
  anything above it.
- **No ignition sources.** Soldering iron unplugged and cold. Nothing hot
  on the bench. Butane is heavier than air and pools low, so keep the floor
  area clear too.
- **Ventilate between every run.** Open a window, lid off, five minutes.
- **Hold the can UPRIGHT. Nozzle up gives vapour, inverted gives liquid.**
  Lighter refill cans have no dip tube, so the valve draws from whichever
  phase is at the top. This is the single most important handling rule here.
- Spray into the air inside the container, never directly at the sensor.
  Liquid butane boils at about -0.5 C and the expanding stream is colder.
  The MQ-6's sensing element is hot ceramic; a cryogenic spray cracks it.

---

## Step 1 — Swap to the logger (10 min)

The board currently runs the demo sketch. For collection, flash
`firmware/sniff_logger/sniff_logger.ino`.

**Before you flash, confirm your load resistor.** Power down, put the meter
across the resistor on the MQ-6 module, and set `RL_KOHM` in the sketch to
what you read. It is set to 10. If that is wrong, every resistance value in
the dataset is wrong and you will not find out until the model refuses to
train.

Open the serial monitor at 115200. You should see CSV rows once a second
with plausible temperature and humidity.

```
python collect.py --list-ports      # find your port name
```

## Step 2 — Check Ro (2 min)

Send `r`. It should print 19.42 kΩ. If it does, skip recalibration.

If you need to redo it: clean room air, away from the kitchen, send `c`. It
averages 60 seconds and saves to flash.

Watch the first few lines. **If `Rs` reads 0.00 the divider is bypassed** and
GPIO34 is seeing the full 5 V rail. That happened once during assembly. Stop
and fix the wiring before recording anything.

## Step 3 — Dosing maths (5 min)

Measure the container's volume. Water bottle method, or length × width ×
height in cm gives you millilitres directly.

```
mL of butane to inject  =  container_mL  ×  target_ppm / 1,000,000
```

For 2,000 ppm in a 5 litre container that is 10 mL.

- Container volume: ________ mL
- mL of VAPOUR for 2,000 ppm: ________ mL

Butane is about twice as dense as air and pools at the bottom. Mount the
sensor at mid-height, wait 30 to 60 s after injecting, and rock the
container gently to mix. Do not lift the lid to check.

**That figure is millilitres of VAPOUR, not liquid.** Liquid butane expands
roughly 240x when it vaporises, so 1 mL of liquid becomes about 240 mL of
gas. Injecting 10 mL of liquid into a 5 L container would release about
2.4 L of butane, roughly half the container, which is around 480,000 ppm
against a lower explosive limit of 1.86%. Overshooting by a factor of 250
is not a measurement error, it is a fire. Keep the can upright and dose
vapour only.

The vapour volume you need is tiny in liquid terms: 2,000 ppm in 5 L is
about 0.04 mL of liquid. That is not a quantity you can measure, which is
the practical reason vapour is the only workable phase.

**With a syringe:** can upright, draw vapour from the headspace, inject
through a small hole in the lid, tape over it.

**Without a syringe:** a 0.5 second burst **from an upright can** into a
5 L container lands roughly in the right range. Your concentrations are
then approximate, so write "nominal, by timed release" in your notes and
say so in the report. Approximate and labelled beats precise and invented.
A burst from an inverted can is liquid and is not a smaller version of the
same thing; it is two orders of magnitude more butane.

---

## Step 4 — Collect (about 2.5 hours)

Eight sessions, ten minutes each. **Two replicates of each condition is not
optional.** Training holds out whole sessions, and with only one session per
condition there is nothing to hold out, so the split falls back to something
much weaker. The second replicate is what makes the numbers defensible.

Run them in this order so the humid setup only has to happen once.

### Round A — dry air

```bash
python collect.py --port YOUR_PORT --gas 0    --rh dry --duration 600 --notes "rep 1"
# inject gas, wait 30 s for it to disperse, then:
python collect.py --port YOUR_PORT --gas 2000 --rh dry --duration 600 --notes "rep 1"
# lid off, ventilate 5 min
python collect.py --port YOUR_PORT --gas 0    --rh dry --duration 600 --notes "rep 2"
python collect.py --port YOUR_PORT --gas 2000 --rh dry --duration 600 --notes "rep 2"
```

### Make it humid (20 min, unattended)

Wet paper towel or sponge in the container, sensor inside, lid on. Leave it
twenty minutes. Watch the serial monitor until RH stops climbing.

Do not let the towel touch the sensor.

### Round B — humid air

```bash
python collect.py --port YOUR_PORT --gas 0    --rh humid --duration 600 --notes "rep 1"
python collect.py --port YOUR_PORT --gas 2000 --rh humid --duration 600 --notes "rep 1"
python collect.py --port YOUR_PORT --gas 0    --rh humid --duration 600 --notes "rep 2"
python collect.py --port YOUR_PORT --gas 2000 --rh humid --duration 600 --notes "rep 2"
```

**`--gas 0 --rh humid` is the most important session you will record.**
Wet air, no gas. That is the false alarm case the entire project exists to
address. If you run short on time, cut a dry replicate, never this one.

You end with ~4,800 labelled rows. That is plenty for a 234 parameter model.

---

## Step 5 — Clear the simulation, then train (10 min)

```bash
python make_synthetic.py --clean     # DO THIS FIRST
python train_tflite.py
```

The `--clean` step matters. If simulated and recorded sessions are both in
`data/`, the metrics describe neither one.

`train_tflite.py` prints the comparison table, writes the two figures to
`results/`, saves `results/metrics_tflite.json`, and drops `model_data.h`
straight into `firmware/sniff_web/`.

Read the VERDICT line at the bottom. It tells you whether the model actually
beat the conventional detector. **If it did not, report that.** A clean
negative result is a real finding and it is far stronger than a claim you
never measured. The script is written to tell you either way.

Check the header of the regenerated `model_data.h`. The `Data source:` line
should now read `recorded`, not `simulated`.

## Step 6 — Reflash and demo (20 min)

Open `firmware/sniff_web/sniff_web.ino` and upload. `model_data.h` is
already beside it.

Set `RL_KOHM` to the same value you used in the logger.

Serial output shows both decisions side by side:

```
rs_ro,temp_c,rh_pct,drs_dt,model,confidence,threshold_says
0.8821,22.40,38.10,-0.00012,safe,0.031,safe
```

### Connecting a phone to the dashboard

Join `SNIFF-Detector`, password `sniffgas2026`, then open
`http://192.168.4.1`. Type the `http://` yourself; without it the phone
searches instead of navigating.

Two things stop the phone associating, and both have bitten this build:

- **Private Wi-Fi Address must be off** for this network. iOS randomises
  the MAC otherwise and the ESP32's small association table rejects it.
- **Turn cellular data off** while demoing. The phone sees a network with no
  internet and silently falls back to cellular, which drops the AP.

The serial log prints `# AP ip=... clients=N hits=N` every 8 seconds.
`clients=0` means the phone never associated at the radio level, so the
problem is the join, not the page. `clients=1 hits=N` with N climbing means
the page is being served and you are fine.

### Before you present, run the self-test

Send `t`. The alarm asserts for 4 seconds: red LED on, buzzer on, and the
log prints

```
# SELF-TEST: alarm asserted for 4 s
# SELF-TEST complete, outputs restored
```

**`m` is a latching toggle.** If you press it to silence the buzzer while
you talk, it stays muted until you press it again. Send `m` and read the
`# muted=` line to see which state you are in. A silent buzzer with a
working LED is almost always this, not a wiring fault.

### The demo that lands

1. Run it in humid air with no gas. The `threshold_says` column should read
   LEAK while `model` reads safe. That one line is the whole thesis, and the
   dashboard shows it without anyone having to read a serial log.
2. Then release gas. Both flip to LEAK, red LED on, buzzer sounds.

---

## If something breaks

| Symptom | Cause |
|---|---|
| `rs_kohm` is nan | Heater has no 5 V. Check the rail. |
| `Rs = 0.00` on every sample | Divider bypassed, GPIO34 seeing ≥2.5 V. Rewire. |
| Serial port busy | Serial Monitor is holding it, or the board is unplugged. |
| Temperature/humidity nan | DHT22 wiring, or GPIO26 is not driven high |
| `rs_ro` near 1.0 and never moves | Ro was calibrated in gassy air. Redo in clean air. |
| Buzzer silent, LED works | Muted. Send `m` and read the `# muted=` line. |
| Page will not load, `clients=0` | Phone never joined. Private Wi-Fi Address, or cellular. |
| Page will not load, `clients=1` | Missing `http://` in the address bar. |
| Model predicts one class always | Classes imbalanced, or gas never reached the sensor. Check that Rs actually dropped during the gas sessions. |
| Training says only one class | A gas session got labelled safe. Check the `--gas` flags. |

---

## Hard truth about the schedule

Data collection is the only thing here that cannot be compressed. The sensor
responds in physical time and humid air takes twenty minutes to equilibrate.
Everything else is minutes.

Without butane there is no leak-class data, and the model stays trained on
a simulation. The demo still runs, but the numbers in the report describe
`make_synthetic.py` rather than a sensor, and that has to be said out loud
in the writeup. That is the one hard dependency. Solve it first.
