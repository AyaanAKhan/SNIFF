/*
 * SNIFF - data collection firmware
 * Author: Ayaan Khan
 *
 * Streams one CSV row per second over serial. collect.py on the laptop
 * attaches session metadata and writes the training file.
 *
 * Also runs the conventional fixed-threshold detector alongside, so every
 * recorded sample already carries what the old method would have decided.
 * That makes the baseline comparison free at training time.
 *
 * Pin map matches the board as actually wired:
 *   GPIO34  MQ-6 AOUT through a 10k/10k divider   (ADC1 - required, see below)
 *   GPIO27  DHT22 data
 *   GPIO26  DHT22 power (held HIGH; the 3V3 pin is covered on this breadboard)
 *   GPIO33  red LED   + 220R
 *   GPIO25  green LED + 220R
 *   GPIO13  active buzzer
 *
 * Serial commands @ 115200:
 *   c  calibrate Ro in clean air (60 s average, saved to flash)
 *   r  print stored Ro
 *   h  reprint the CSV header
 */

#include <DHT.h>
#include <Preferences.h>

#define MQ_PIN      34
#define DHT_PIN     27
#define DHT_PWR     26
#define LED_ALARM   33
#define LED_SAFE    25
#define BUZZER_PIN  13

// Supply to the MQ-6 sensing circuit.
const float VC = 5.0;

// Load resistance on the MQ-6 module, in kilohms.
// MEASURE YOURS. Modules ship with anything from 1k to 20k and every
// resistance value downstream is wrong if this number is wrong.
const float RL_KOHM = 10.0;

// Two 10k resistors: the ADC sees half of AOUT.
const float DIVIDER_RATIO = 2.0;

const int ADC_SAMPLES = 32;
const int SLOPE_WINDOW = 10;              // 10 s of history for dRs/dt
const unsigned long LOOP_MS = 1000;
const unsigned long DHT_MIN_INTERVAL_MS = 2000;   // DHT22 caps at 0.5 Hz

// Conventional detector: alarm when Rs/Ro falls below this. Provisional.
const float BASELINE_THRESHOLD = 0.45;

DHT dht(DHT_PIN, DHT22);
Preferences prefs;

float Ro_kohm = 0.0;
float rsWindow[SLOPE_WINDOW];
int wIdx = 0;
bool wFull = false;

float lastTemp = NAN, lastRH = NAN;
unsigned long lastDht = 0, lastLoop = 0;

float readAoutVolts() {
  uint32_t acc = 0;
  for (int i = 0; i < ADC_SAMPLES; i++) {
    acc += analogReadMilliVolts(MQ_PIN);   // uses the chip's factory ADC calibration
    delayMicroseconds(200);
  }
  return ((acc / (float)ADC_SAMPLES) / 1000.0) * DIVIDER_RATIO;
}

float computeRs(float vout) {
  if (vout <= 0.01) return NAN;           // heater unpowered or open circuit
  if (vout >= VC - 0.01) return 0.0;      // rail
  return RL_KOHM * (VC - vout) / vout;
}

float computeSlope(float rs) {
  rsWindow[wIdx] = rs;
  int oldest = (wIdx + 1) % SLOPE_WINDOW;
  wIdx = (wIdx + 1) % SLOPE_WINDOW;
  if (wIdx == 0) wFull = true;
  if (!wFull) return 0.0;
  return (rs - rsWindow[oldest]) / (float)(SLOPE_WINDOW - 1);   // kohm/s
}

void printHeader() {
  Serial.println(F("# millis,aout_v,rs_kohm,rs_ro,temp_c,rh_pct,drs_dt,baseline_alarm"));
}

void calibrateRo() {
  Serial.println(F("# calibrating Ro. Clean air, 60 s. Do not breathe on the sensor."));
  float acc = 0; int n = 0;
  for (int i = 0; i < 60; i++) {
    float rs = computeRs(readAoutVolts());
    if (!isnan(rs) && rs > 0) { acc += rs; n++; }
    Serial.printf("# %2d/60  Rs = %.2f kohm\n", i + 1, rs);
    delay(1000);
  }
  if (n == 0) { Serial.println(F("# FAILED: no valid readings. Check the 5 V heater rail.")); return; }
  Ro_kohm = acc / n;
  prefs.begin("sniff", false); prefs.putFloat("ro", Ro_kohm); prefs.end();
  Serial.printf("# Ro = %.2f kohm, saved\n", Ro_kohm);
}

void setup() {
  Serial.begin(115200);
  delay(400);

  // GPIO26 powers the DHT22 because 3V3 is physically covered on this board.
  // The DHT22 draws ~1.5 mA, well inside the 40 mA per-pin limit.
  pinMode(DHT_PWR, OUTPUT);
  digitalWrite(DHT_PWR, HIGH);
  delay(100);

  pinMode(LED_ALARM, OUTPUT);
  pinMode(LED_SAFE, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_ALARM, LOW);
  digitalWrite(LED_SAFE, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // 11 dB attenuation gives roughly 0-3.1 V. The divider keeps the 5 V
  // sensor swing at 2.5 V max, inside the linear part of that range.
  analogSetPinAttenuation(MQ_PIN, ADC_11db);

  dht.begin();

  prefs.begin("sniff", true);
  Ro_kohm = prefs.getFloat("ro", 0.0);
  prefs.end();

  for (int i = 0; i < SLOPE_WINDOW; i++) rsWindow[i] = 0.0;

  Serial.println(F("# SNIFF logger"));
  if (Ro_kohm <= 0) Serial.println(F("# WARNING: Ro not calibrated. Send 'c' in clean air."));
  else Serial.printf("# Ro = %.2f kohm\n", Ro_kohm);
  Serial.printf("# RL = %.1f kohm (verify against your module)\n", RL_KOHM);
  printHeader();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') calibrateRo();
    else if (c == 'r') Serial.printf("# Ro = %.2f kohm\n", Ro_kohm);
    else if (c == 'h') printHeader();
  }

  unsigned long now = millis();
  if (now - lastLoop < LOOP_MS) return;
  lastLoop = now;

  // The DHT22 only updates every 2 s; hold the last good reading between.
  if (now - lastDht >= DHT_MIN_INTERVAL_MS) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) lastTemp = t;
    if (!isnan(h)) lastRH = h;
    lastDht = now;
  }

  float vout  = readAoutVolts();
  float rs    = computeRs(vout);
  float rsRo  = (Ro_kohm > 0 && !isnan(rs)) ? (rs / Ro_kohm) : NAN;
  float slope = isnan(rs) ? 0.0 : computeSlope(rs);

  bool alarm = (!isnan(rsRo) && rsRo < BASELINE_THRESHOLD);

  // LEDs give a visual of what the conventional detector would have decided.
  // The buzzer is deliberately NOT driven here. This firmware records data for
  // ten minutes at a stretch, often with gas present on purpose, and a logger
  // that alarms through every gas session is unusable. The alarm output lives
  // in sniff_web, which is the firmware that actually has to behave like a
  // detector.
  digitalWrite(LED_ALARM, alarm ? HIGH : LOW);
  digitalWrite(LED_SAFE,  alarm ? LOW  : HIGH);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.printf("%lu,%.4f,%.3f,%.4f,%.2f,%.2f,%.5f,%d\n",
                now, vout, rs, rsRo, lastTemp, lastRH, slope, alarm ? 1 : 0);
}
