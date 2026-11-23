/*
 * SNIFF - inference firmware
 * Author: Ayaan Khan
 *
 * Runs the trained classifier on the ESP32 and drives the alarm.
 * Also runs the conventional fixed-threshold detector in parallel and
 * prints both decisions, so a live demo shows the difference directly.
 *
 * REQUIRES model.h, which train.py writes into this folder. Train first.
 *
 * Pin map (as wired):
 *   GPIO34  MQ-6 AOUT through 10k/10k divider   (ADC1; ADC2 dies with WiFi)
 *   GPIO27  DHT22 data
 *   GPIO26  DHT22 power, held HIGH
 *   GPIO33  red LED   + 220R
 *   GPIO25  green LED + 220R
 *   GPIO13  active buzzer
 *
 * Serial @ 115200:
 *   c  recalibrate Ro in clean air
 *   r  print Ro
 *   m  toggle mute (silences the buzzer, LEDs keep working)
 *
 * Why a hand-written forward pass instead of TensorFlow Lite Micro:
 * this network is 4-16-8-2, about 234 parameters and roughly 250 multiply
 * accumulates. The interpreter would add tens of kilobytes of flash and a
 * build dependency to run arithmetic that fits in thirty lines. The maths
 * is identical. train.py reports what int8 quantization would cost, so the
 * quantization claim is backed by a measurement either way.
 */

#include <DHT.h>
#include <Preferences.h>
#include "model.h"

#define MQ_PIN      34
#define DHT_PIN     27
#define DHT_PWR     26
#define LED_ALARM   33
#define LED_SAFE    25
#define BUZZER_PIN  13

const float VC = 5.0;
const float RL_KOHM = 10.0;          // MEASURE YOURS, same value as in the logger
const float DIVIDER_RATIO = 2.0;

const int ADC_SAMPLES = 32;
const int SLOPE_WINDOW = 10;
const unsigned long LOOP_MS = 1000;
const unsigned long DHT_MIN_INTERVAL_MS = 2000;

// Conventional detector, for the side-by-side comparison.
const float BASELINE_THRESHOLD = 0.45;

// Require this many consecutive leak predictions before the alarm latches.
// One noisy sample should not set off a gas alarm.
const int CONFIRM_SAMPLES = 3;

DHT dht(DHT_PIN, DHT22);
Preferences prefs;

float Ro_kohm = 0.0;
float rsWindow[SLOPE_WINDOW];
int wIdx = 0;
bool wFull = false;

float lastTemp = NAN, lastRH = NAN;
unsigned long lastDht = 0, lastLoop = 0;
int leakStreak = 0;
bool muted = false;

// ------------------------------------------------------------ sensing

float readAoutVolts() {
  uint32_t acc = 0;
  for (int i = 0; i < ADC_SAMPLES; i++) {
    acc += analogReadMilliVolts(MQ_PIN);
    delayMicroseconds(200);
  }
  return ((acc / (float)ADC_SAMPLES) / 1000.0) * DIVIDER_RATIO;
}

float computeRs(float vout) {
  if (vout <= 0.01) return NAN;
  if (vout >= VC - 0.01) return 0.0;
  return RL_KOHM * (VC - vout) / vout;
}

float computeSlope(float rs) {
  rsWindow[wIdx] = rs;
  int oldest = (wIdx + 1) % SLOPE_WINDOW;
  wIdx = (wIdx + 1) % SLOPE_WINDOW;
  if (wIdx == 0) wFull = true;
  if (!wFull) return 0.0;
  return (rs - rsWindow[oldest]) / (float)(SLOPE_WINDOW - 1);
}

// ------------------------------------------------------------ model

// Scratch buffers sized to the widest layer in the network.
static float bufA[32];
static float bufB[32];

static void denseRelu(const float *in, int nIn,
                      const float *W, const float *b,
                      int nOut, float *out, bool relu) {
  for (int j = 0; j < nOut; j++) {
    float acc = b[j];
    for (int i = 0; i < nIn; i++) acc += in[i] * W[i * nOut + j];
    out[j] = (relu && acc < 0.0f) ? 0.0f : acc;
  }
}

/* Returns true for "leak". conf gets the probability of leak.
 *
 * scikit-learn's MLPClassifier emits ONE logit for a binary problem and
 * applies a sigmoid, not two logits with a softmax. L2_OUT tells us which
 * shape we got, so both are handled and the firmware matches whatever
 * train.py produced. */
static bool runModel(const float *feat, float *conf) {
  // Normalise with the constants frozen at training time.
  for (int i = 0; i < N_FEATURES; i++)
    bufA[i] = (feat[i] - FEAT_MEAN[i]) / FEAT_STD[i];

  denseRelu(bufA, L0_IN, &L0_W[0][0], L0_B, L0_OUT, bufB, true);
  denseRelu(bufB, L1_IN, &L1_W[0][0], L1_B, L1_OUT, bufA, true);
  denseRelu(bufA, L2_IN, &L2_W[0][0], L2_B, L2_OUT, bufB, false);

  if (L2_OUT == 1) {
    *conf = 1.0f / (1.0f + expf(-bufB[0]));      // sigmoid
    return bufB[0] > 0.0f;
  }

  // Two-logit softmax. Subtract the max for numerical safety.
  float m = (bufB[0] > bufB[1]) ? bufB[0] : bufB[1];
  float e0 = expf(bufB[0] - m), e1 = expf(bufB[1] - m);
  *conf = e1 / (e0 + e1);
  return bufB[1] > bufB[0];
}

// ------------------------------------------------------------ calibration

void calibrateRo() {
  Serial.println(F("# calibrating Ro. Clean air, 60 s."));
  float acc = 0; int n = 0;
  for (int i = 0; i < 60; i++) {
    float rs = computeRs(readAoutVolts());
    if (!isnan(rs) && rs > 0) { acc += rs; n++; }
    Serial.printf("# %2d/60  Rs = %.2f kohm\n", i + 1, rs);
    delay(1000);
  }
  if (n == 0) { Serial.println(F("# FAILED: check the 5 V heater rail.")); return; }
  Ro_kohm = acc / n;
  prefs.begin("sniff", false); prefs.putFloat("ro", Ro_kohm); prefs.end();
  Serial.printf("# Ro = %.2f kohm, saved\n", Ro_kohm);
}

// ------------------------------------------------------------ setup / loop

void setup() {
  Serial.begin(115200);
  delay(400);

  pinMode(DHT_PWR, OUTPUT);
  digitalWrite(DHT_PWR, HIGH);
  delay(100);

  pinMode(LED_ALARM, OUTPUT);
  pinMode(LED_SAFE, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_ALARM, LOW);
  digitalWrite(LED_SAFE, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  analogSetPinAttenuation(MQ_PIN, ADC_11db);
  dht.begin();

  prefs.begin("sniff", true);
  Ro_kohm = prefs.getFloat("ro", 0.0);
  prefs.end();

  for (int i = 0; i < SLOPE_WINDOW; i++) rsWindow[i] = 0.0;

  Serial.println(F("# SNIFF inference"));
  Serial.printf("# model: %d features, %d layers\n", N_FEATURES, N_LAYERS);
  if (Ro_kohm <= 0) Serial.println(F("# WARNING: Ro not calibrated. Send 'c'."));
  else Serial.printf("# Ro = %.2f kohm\n", Ro_kohm);
  Serial.println(F("# rs_ro,temp_c,rh_pct,drs_dt,model,confidence,threshold_says"));
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') calibrateRo();
    else if (c == 'r') Serial.printf("# Ro = %.2f kohm\n", Ro_kohm);
    else if (c == 'm') { muted = !muted; Serial.printf("# muted=%d\n", muted); }
  }

  unsigned long now = millis();
  if (now - lastLoop < LOOP_MS) return;
  lastLoop = now;

  if (now - lastDht >= DHT_MIN_INTERVAL_MS) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) lastTemp = t;
    if (!isnan(h)) lastRH = h;
    lastDht = now;
  }

  float rs = computeRs(readAoutVolts());
  if (isnan(rs) || Ro_kohm <= 0 || isnan(lastTemp) || isnan(lastRH)) {
    Serial.println(F("# waiting for valid sensor data"));
    return;
  }

  float rsRo  = rs / Ro_kohm;
  float slope = computeSlope(rs);

  float feat[N_FEATURES] = { rsRo, lastTemp, lastRH, slope };
  float conf = 0.0f;
  bool modelLeak = runModel(feat, &conf);

  // Debounce: one sample should never trip a gas alarm.
  leakStreak = modelLeak ? (leakStreak + 1) : 0;
  bool alarm = (leakStreak >= CONFIRM_SAMPLES);

  bool thresholdSays = (rsRo < BASELINE_THRESHOLD);

  digitalWrite(LED_ALARM, alarm ? HIGH : LOW);
  digitalWrite(LED_SAFE,  alarm ? LOW  : HIGH);
  digitalWrite(BUZZER_PIN, (alarm && !muted) ? HIGH : LOW);

  Serial.printf("%.4f,%.2f,%.2f,%.5f,%s,%.3f,%s\n",
                rsRo, lastTemp, lastRH, slope,
                alarm ? "LEAK" : "safe", conf,
                thresholdSays ? "LEAK" : "safe");
}
