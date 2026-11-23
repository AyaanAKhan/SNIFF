# SNIFF — hardware reference

Ayaan Khan

Everything below was measured on the assembled unit unless marked otherwise.

## Pin map, as physically wired

| ESP32 pin | Connects to | Notes |
|---|---|---|
| GPIO34 | MQ-6 AOUT via 10k/10k divider | ADC1. Input only, no pullups. |
| GPIO27 | DHT22 DATA | |
| GPIO26 | DHT22 VCC, driven HIGH in firmware | The 3V3 header pin is physically obstructed on this breadboard |
| GPIO33 | Red LED anode, 220 Ω to ground | Alarm. Currently not lighting, see Known issues |
| GPIO25 | Green LED anode, 220 Ω to ground | Safe |
| GPIO13 | Active buzzer + | Driven directly, no transistor |
| VIN (5V) | MQ-6 VCC (heater + sense) | From USB |
| GND | Common ground rail | |

## Parts

| Part | Detail |
|---|---|
| MCU | ESP32 Dev Module, CP210x/CH340 USB serial, enumerates as `/dev/cu.usbserial-0001` |
| Gas sensor | MQ-6, tin dioxide (SnO₂), module with onboard load resistor |
| Temp/humidity | DHT22 (AM2302) |
| Divider | 2 × 10 kΩ, 1% |
| LED resistors | 2 × 220 Ω |
| Buzzer | Active, 3–24 V |
| Battery (not yet integrated) | 18650, 2600 mAh, with TP4056 charger and MT3608 boost |

## Two wiring decisions that are not arbitrary

**The divider is required.** The MQ-6 module's analog output swings across the
full 0–5 V supply. The ESP32 ADC pins are rated to 3.3 V. Two 10 kΩ resistors
halve it, which protects the pin and keeps the signal inside the more linear
region of the converter.

Confirmed in practice: an early calibration run returned `Rs = 0.00 kohm` on
all sixty samples because the divider had been bypassed and the pin was
saturated at the rail.

**GPIO34 is on ADC1, and has to be.** ADC2 is claimed by the WiFi driver and
stops returning valid conversions once that driver starts. This design runs a
SoftAP for the dashboard, so the gas channel cannot live on ADC2. The failure
mode would be silent.

## Measured electrical values

| Quantity | Value | How obtained |
|---|---|---|
| Ro, clean-air baseline | **19.42 kΩ** | 60 s average via firmware `c` command, stored in NVS |
| RL, load resistor | 10.0 kΩ assumed | Not independently metered. Cancels out of the Rs/Ro ratio, so it affects only the absolute kΩ column |
| VC, sensor supply | 5.0 V | USB |
| Divider ratio | 2.0 | 10k/10k |

Rs is computed as `RL × (VC − Vout) / Vout`, where Vout is the divider output
multiplied back up by 2.0. Each sample is a 32-reading average via
`analogReadMilliVolts`.

## Firmware build, measured from the compiler

| | |
|---|---|
| Flash | 1,016,496 bytes, 77% of partition |
| RAM (static) | 65,864 bytes, 20% |
| Model flatbuffer | 3,440 bytes |
| Tensor arena | 16 KB allocated |

## Model

| | |
|---|---|
| Architecture | 4 → 16 → 8 → 2, ReLU hidden, softmax out |
| Parameters | 234 |
| Inputs | rs_ro, temp_c, rh_pct, drs_dt (10 s window) |
| Runtime | TensorFlow Lite Micro, `MicroMutableOpResolver<4>` |
| Ops registered | FullyConnected, Softmax, Quantize, Dequantize |
| Quantization | Full int8, representative dataset from the training split |
| Quantization cost | +0.00 percentage points vs float32 |

## Wi-Fi dashboard

| | |
|---|---|
| Mode | SoftAP, no router, no internet |
| SSID | `SNIFF-Detector` |
| Password | `sniffgas2026` |
| Address | `http://192.168.4.1` |
| Endpoints | `/` page from PROGMEM, `/d` JSON poll |

The page carries no external resources, because a client joined to a network
with no internet cannot fetch a CDN.

## Serial interface, 115200 baud

| Key | Effect |
|---|---|
| `c` | Recalibrate Ro, 60 s average in clean air, saves to NVS |
| `r` | Print stored Ro |
| `m` | Toggle mute. **Latches** — stays muted until pressed again |
| `t` | Alarm self-test, asserts outputs for 4 s |

`sniff_logger` CSV columns:
`millis,aout_v,rs_kohm,rs_ro,temp_c,rh_pct,drs_dt,baseline_alarm`

`sniff_web` line:
`rs_ro,temp_c,rh_pct,drs_dt,model,confidence,threshold_says`

## Power-on warm-up lockout

The MQ-6 sensing element must reach operating temperature before Rs means
anything. Cold, its resistance sits far below Ro, so rs_ro looks like a large
gas concentration and both the model and the fixed threshold call a leak.
Measured on this unit: rs_ro ≈ 0.40 at power-on, climbing past 1.4 over tens
of minutes.

`sniff_web` therefore suppresses the alarm outputs for the first **90 seconds**
after boot. Readings still compute, stream to serial and serve to the
dashboard during that window; only the LED and buzzer are held off. A
countdown is printed each second.

90 s stops the boot transient. It does not substitute for proper warm-up
before taking measurements.

## Observed sensor behaviour

| Observation | Value |
|---|---|
| Humidity effect, no gas, 51.3% → 76.4% RH | rs_ro 1.590 → 1.345, −15.4% |
| Temperature coefficient, fitted from recorded data | −0.03012 rs_ro per °C |
| Humidity coefficient, fitted | +0.00539 rs_ro per %RH |
| Residual preheat drift over 75 min, after T/RH correction | ≈ +0.04, about 3% of baseline |
| Gas response, as dosed on 10 Nov | rs_ro 1.590 → 0.096, −94% (sensor saturated) |

The MQ-6 datasheet specifies 24 hours of preheat before calibration. The
drift observed here is consistent with that being a real requirement.

## Known issues

**Red LED on GPIO33 does not light.** The firmware drives it on the line
immediately before the buzzer, and the buzzer works, so the fault is the LED
itself, its polarity, or its breadboard row. Cosmetic only; the dashboard and
serial output both carry alarm state.

**Battery operation not integrated.** The MT3608 boost converter must be
trimmed to 5.0 V with a 5/64" flathead, with nothing connected to its output
until the meter reads 5.0 V. The module can deliver up to 28 V and the
ESP32 board's regulator is rated to roughly 15 V.

**RL not independently metered.** Assumed 10.0 kΩ. Affects the absolute
`rs_kohm` column only, not `rs_ro` and not any model input.
