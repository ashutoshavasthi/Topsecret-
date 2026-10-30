// SkyMind Nano: ESP32-CAM sensor head
// ------------------------------------
// * Streams OV2640 video as MJPEG:   http://<this board's IP>:81/stream
// * Reads 3 x VL53L0X (left -35 deg, centre 0 deg, right +35 deg) at ~20 Hz and
//   broadcasts them as UDP JSON on port 2399: {"t":123456,"tof_mm":[812,1500,64]}
//
// Board: "AI Thinker ESP32-CAM" (Arduino-ESP32 core 2.x or 3.x)
// Library: "VL53L0X" by Pololu (Library Manager)
//
// Wiring (the camera uses most pins, these are the free ones when the SD card is unused):
//   I2C SDA -> GPIO14, SCL -> GPIO15, 3.3 V and GND to all three sensors
//   XSHUT of LEFT sensor  -> GPIO2   (disconnect while flashing if upload fails)
//   XSHUT of RIGHT sensor -> GPIO13
//   CENTRE sensor XSHUT not connected (always on)
// Do NOT use GPIO12 (boot strapping pin; the breakout's XSHUT pull-up would stop the board booting).
//
// The board joins the ESP-Drone's WiFi access point, so the laptop sees both on one network.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <VL53L0X.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// ---- settings -------------------------------------------------------------------------
static const char *WIFI_SSID = "ESP-DRONE_XXXXXXXXXXXX";  // the ESP-Drone's AP name
static const char *WIFI_PASS = "12345678";                // ESP-Drone default password
static const uint16_t UDP_PORT = 2399;
static const uint32_t TOF_PERIOD_MS = 50;  // 20 Hz
static const uint32_t TIMING_BUDGET_US = 33000;

// ---- AI-Thinker ESP32-CAM camera pins --------------------------------------------------
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ---- ToF ----------------------------------------------------------------------------------
static const int I2C_SDA = 14;
static const int I2C_SCL = 15;
static const int XSHUT_LEFT = 2;
static const int XSHUT_RIGHT = 13;

TwoWire tofWire = TwoWire(1);  // camera SCCB uses I2C port 0
VL53L0X tofLeft, tofCentre, tofRight;
WiFiUDP udp;
httpd_handle_t streamServer = nullptr;

static bool initSensor(VL53L0X &s, uint8_t address) {
  s.setBus(&tofWire);
  s.setTimeout(100);
  if (!s.init()) return false;
  s.setAddress(address);
  // Long-range mode: lower signal limit, wider VCSEL pulses (Pololu "LONG_RANGE" example).
  s.setSignalRateLimit(0.1);
  s.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
  s.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
  s.setMeasurementTimingBudget(TIMING_BUDGET_US);
  s.startContinuous();
  return true;
}

static void initToF() {
  pinMode(XSHUT_LEFT, OUTPUT);
  pinMode(XSHUT_RIGHT, OUTPUT);
  digitalWrite(XSHUT_LEFT, LOW);  // hold left and right in reset
  digitalWrite(XSHUT_RIGHT, LOW);
  delay(10);
  tofWire.begin(I2C_SDA, I2C_SCL, 400000);
  // Only the centre sensor is awake at 0x29: move it, then wake the others one by one.
  if (!initSensor(tofCentre, 0x30)) Serial.println("centre VL53L0X not found");
  digitalWrite(XSHUT_LEFT, HIGH);
  delay(10);
  if (!initSensor(tofLeft, 0x31)) Serial.println("left VL53L0X not found");
  digitalWrite(XSHUT_RIGHT, HIGH);
  delay(10);
  if (!initSensor(tofRight, 0x32)) Serial.println("right VL53L0X not found");
}

static uint16_t readMm(VL53L0X &s) {
  uint16_t mm = s.readRangeContinuousMillimeters();
  if (s.timeoutOccurred()) return 0;  // 0 = no data; the laptop treats it as "no return"
  return mm;                          // 8190 / 65535 = nothing in range
}

// ---- camera + MJPEG stream -------------------------------------------------------------
static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_QVGA;  // 320 x 240, matches config/nano.yaml
  c.jpeg_quality = 12;
  c.fb_count = 2;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always send the newest frame (low latency)
  c.fb_location = CAMERA_FB_IN_PSRAM;
  return esp_camera_init(&c) == ESP_OK;
}

#define PART_BOUNDARY "skymindframe"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t streamHandler(httpd_req_t *req) {
  char header[64];
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  while (res == ESP_OK) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
      break;
    }
    size_t n = snprintf(header, sizeof(header), STREAM_PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, header, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    delay(100);  // ~10 fps is plenty for the AI and saves WiFi airtime for flight control
  }
  return res;
}

static void startStream() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 81;
  cfg.ctrl_port = 32769;
  httpd_uri_t uri = {.uri = "/stream", .method = HTTP_GET, .handler = streamHandler, .user_ctx = nullptr};
  if (httpd_start(&streamServer, &cfg) == ESP_OK) httpd_register_uri_handler(streamServer, &uri);
}

// ---- main ------------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  if (!initCamera()) Serial.println("camera init failed");
  initToF();
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // lower, steadier latency
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.printf("\nIP %s  stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str(),
                WiFi.localIP().toString().c_str());
  startStream();
}

void loop() {
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last < TOF_PERIOD_MS) {
    delay(1);
    return;
  }
  last = now;
  // Order must match sensors.tof_headings_deg in config/nano.yaml: [-35, 0, 35].
  // SkyMind angles are counter-clockwise, so -35 deg is the RIGHT-hand sensor.
  char msg[96];
  int n = snprintf(msg, sizeof(msg), "{\"t\":%lu,\"tof_mm\":[%u,%u,%u]}", (unsigned long)now, readMm(tofRight),
                   readMm(tofCentre), readMm(tofLeft));
  if (WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(WiFi.broadcastIP(), UDP_PORT);
    udp.write((const uint8_t *)msg, n);
    udp.endPacket();
  }
}
