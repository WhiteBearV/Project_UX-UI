#include <Servo.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

#define LED_PIN    6
#define LED_COUNT  24
#define ENABLE_PIN 2

// ✅ BUZZER PIN (โมดูล 3 ขา: VCC/GND/SIG)
#define BUZZER_PIN 7

static const uint8_t N = 5;
const uint8_t SERVO_PIN[N] = {3, 5, 9, 10, 11};

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
uint8_t brightnessVal = 80;

// ✅ เพิ่ม RAINBOW + POLICE (ไม่มี BLINK)
enum LedMode : uint8_t {
  MODE_MANUAL=0,
  MODE_WHITE,
  MODE_RED,
  MODE_SUNRISE,
  MODE_BREATH,
  MODE_RAINBOW,
  MODE_POLICE,
  MODE_OFF
};

LedMode ledMode = MODE_OFF;
bool ledDirty = true;
unsigned long lastLedTick = 0;
const uint16_t ledTickMs = 35;

uint8_t sunriseOffset = 0;
int breathB = 0; int breathDir = 1;

// ✅ สำหรับ RAINBOW / POLICE
uint8_t rainbowOffset = 0;
bool policeFlip = false;
unsigned long lastPoliceFlip = 0;
const uint16_t policeFlipMs = 250; // สลับแดง/น้ำเงินทุก 250ms

Servo sv[N];
float curA[N] = {0,0,0,0,0};
float tgtA[N] = {0,0,0,0,0};

bool holdAll = true;
bool scanOn = false;
float scanMin=30, scanMax=150, scanSpeedDegPerSec=10;
int scanDir = 1;

// ล็อก S2-S5
bool bookHold = false;
float bookLockPose[4] = {90,90,90,90};

unsigned long lastMoveTick = 0;
const uint16_t moveTickMs = 10;
float moveMaxSpeedDegPerSec = 60.0f;

String line;
bool armed = false;

// debounce ปุ่ม
bool lastRaw=false, stable=false;
unsigned long lastDebounceMs=0;
const unsigned long debounceMs=35;

// --- SHUTDOWN ---
bool shutdownPending = false;

// ---------- utils ----------
static float clampf(float v, float lo, float hi) { return (v<lo)?lo:(v>hi?hi:v); }

// ===================== BUZZER (3-pin active module) =====================
// โมดูลบัซเซอร์ 3 ขาส่วนใหญ่ = Active buzzer (สั่ง HIGH/LOW ที่ SIG)
#define BUZZER_IS_ACTIVE   1      // 1=Active (HIGH/LOW), 0=Passive (tone)
#define BUZZER_ACTIVE_LOW  0      // ถ้าไม่ดังให้ลองเปลี่ยนเป็น 1

static bool buzOn = false;
static unsigned long buzOffAt = 0;

static void buzzerOffNow() {
  digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_LOW ? HIGH : LOW);
  buzOn = false;
}

static void buzzerOnNow() {
  digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_LOW ? LOW : HIGH);
  buzOn = true;
}

static void buzzerStart_ms(int dur_ms, int freq_hz) {
  dur_ms = (int)clampf(dur_ms, 10, 2000);
  freq_hz = (int)clampf(freq_hz, 50, 8000);

#if BUZZER_IS_ACTIVE
  buzzerOnNow();
  buzOffAt = millis() + (unsigned long)dur_ms;
#else
  tone(BUZZER_PIN, freq_hz, dur_ms);
#endif
}

static void buzzerUpdate() {
#if BUZZER_IS_ACTIVE
  if (buzOn && (long)(millis() - buzOffAt) >= 0) {
    buzzerOffNow();
  }
#endif
}
// ========================================================================

static uint32_t sunriseColor(uint8_t p) {
  if(p<86) return strip.Color(255, p*2, 0);
  else if(p<171) return strip.Color(255, 170+(p-86), 0);
  else return strip.Color(255, 255, (p-171)*2);
}

static void setAll(uint8_t r, uint8_t g, uint8_t b) {
  strip.fill(strip.Color(r,g,b),0,LED_COUNT);
  strip.show();
}

static void setLedMode(LedMode m) { ledMode=m; ledDirty=true; }

// wheel แบบง่ายสำหรับ RAINBOW
static uint32_t wheel(uint8_t p){
  uint8_t r,g,b;
  if(p < 85){
    r = 255 - p*3; g = p*3; b = 0;
  } else if(p < 170){
    p -= 85;
    r = 0; g = 255 - p*3; b = p*3;
  } else {
    p -= 170;
    r = p*3; g = 0; b = 255 - p*3;
  }
  return strip.Color(r,g,b);
}

static void applyLedMode() {
  unsigned long now = millis();

  // โหมดที่เป็น “คงที่/ครั้งเดียว”
  if (ledMode==MODE_WHITE || ledMode==MODE_RED || ledMode==MODE_OFF || ledMode==MODE_MANUAL) {
    if(!ledDirty) return;
    ledDirty=false;
    strip.setBrightness(brightnessVal);
    if(ledMode==MODE_WHITE) setAll(255,255,255);
    else if(ledMode==MODE_RED) setAll(255,0,0);
    else if(ledMode!=MODE_MANUAL) { strip.clear(); strip.show(); }
    return;
  }

  // โหมดที่เป็น “แอนิเมชัน”
  if (now - lastLedTick < ledTickMs) return;
  lastLedTick = now;

  if (ledMode==MODE_SUNRISE) {
    strip.setBrightness(brightnessVal);
    for(int i=0; i<LED_COUNT; i++) {
      strip.setPixelColor(i, sunriseColor((uint8_t)(sunriseOffset+i*(256/LED_COUNT))));
    }
    strip.show();
    sunriseOffset++;

  } else if (ledMode==MODE_BREATH) {
    breathB += breathDir*3;
    if(breathB>=brightnessVal){breathB=brightnessVal; breathDir=-1;}
    if(breathB<=5){breathB=5; breathDir=1;}
    strip.setBrightness((uint8_t)breathB);
    strip.fill(strip.Color(255,255,255),0,LED_COUNT);
    strip.show();

  } else if (ledMode==MODE_RAINBOW) {
    strip.setBrightness(brightnessVal);
    for(int i=0;i<LED_COUNT;i++){
      uint8_t p = (uint8_t)(rainbowOffset + i*(256/LED_COUNT));
      strip.setPixelColor(i, wheel(p));
    }
    strip.show();
    rainbowOffset++;

  } else if (ledMode==MODE_POLICE) {
    strip.setBrightness(brightnessVal);

    // สลับแดง/น้ำเงินตามเวลา
    if(now - lastPoliceFlip >= policeFlipMs){
      lastPoliceFlip = now;
      policeFlip = !policeFlip;
    }

    int half = LED_COUNT/2;
    uint32_t c1 = policeFlip ? strip.Color(255,0,0) : strip.Color(0,0,255);
    uint32_t c2 = policeFlip ? strip.Color(0,0,255) : strip.Color(255,0,0);

    for(int i=0;i<half;i++) strip.setPixelColor(i, c1);
    for(int i=half;i<LED_COUNT;i++) strip.setPixelColor(i, c2);
    strip.show();
  }
}

static void attachAll() {
  for(int i=0;i<N;i++){
    if(!sv[i].attached()) sv[i].attach(SERVO_PIN[i]);
  }
}

static void detachAll() {
  for(int i=0;i<N;i++){
    if(sv[i].attached()) sv[i].detach();
  }
}

static void applyServoWrite() {
  for(int i=0; i<N; i++){
    if(sv[i].attached()) sv[i].write((int)clampf(curA[i],0,180));
  }
}

static bool nearAll(float target, float eps=0.8f) {
  for(int i=0;i<N;i++){
    if (fabsf(curA[i]-target) > eps) return false;
  }
  return true;
}

static void enterArmedNow() {
  shutdownPending=false;
  armed=true;
  attachAll();
  for(int i=0;i<N;i++){
    curA[i]=0; tgtA[i]=0;
    sv[i].write(0);
  }
  holdAll=true;
  scanOn=false;
  bookHold=false;
  setLedMode(MODE_OFF);
  Serial.println("EVENT ARMED");
}

static void enterSafeNow() {
  shutdownPending=false;
  armed=false;
  holdAll=true;
  scanOn=false;
  bookHold=false;
  setLedMode(MODE_OFF);
  detachAll();
  Serial.println("EVENT SAFE");
}

static void beginShutdown() {
  shutdownPending=true;

  scanOn=false;
  bookHold=false;
  holdAll=false;

  for(int i=0;i<N;i++) tgtA[i]=0;

  moveMaxSpeedDegPerSec = 180.0f;

  // ระหว่างกลับบ้านให้ไฟติด (WHITE)
  setLedMode(MODE_WHITE);

  Serial.println("EVENT SHUTDOWN_BEGIN");
}

static void finishShutdown() {
  holdAll=true;
  scanOn=false;
  bookHold=false;

  setLedMode(MODE_OFF);

  detachAll();
  armed=false;
  shutdownPending=false;

  Serial.println("EVENT SHUTDOWN_DONE");
}

static void updateMotion() {
  if(!armed) return;

  unsigned long now = millis();
  if (now - lastMoveTick < moveTickMs) return;
  float dt = (now - lastMoveTick)/1000.0f;
  lastMoveTick = now;

  if (bookHold) {
    for(int i=1;i<N;i++) tgtA[i] = bookLockPose[i-1];
  }

  if (scanOn && !holdAll) {
    float next = curA[0] + scanDir * scanSpeedDegPerSec * dt;
    if(next>=scanMax){ next=scanMax; scanDir=-1; }
    if(next<=scanMin){ next=scanMin; scanDir=1; }
    tgtA[0] = next;
  }

  for(int i=0;i<N;i++){
    if(holdAll) continue;

    float diff = tgtA[i] - curA[i];

    float maxStep;
    if(i==0 && scanOn) maxStep = scanSpeedDegPerSec * dt;
    else              maxStep = moveMaxSpeedDegPerSec * dt;

    curA[i] += clampf(diff, -maxStep, maxStep);
  }

  applyServoWrite();
}

static void printStatus() {
  Serial.print("STATUS ARMED=");
  Serial.print(armed?1:0);
  Serial.print(" SHUT=");
  Serial.print(shutdownPending?1:0);
  Serial.print(" SCAN=");
  Serial.print(scanOn?1:0);
  Serial.print(" BOOK=");
  Serial.print(bookHold?1:0);
  Serial.print(" HOLD=");
  Serial.print(holdAll?1:0);
  Serial.print(" CUR=");
  for(int i=0;i<N;i++){
    Serial.print((int)curA[i]);
    if(i<N-1) Serial.print(",");
  }
  Serial.println();
}

static void handleCommand(String s) {
  s.trim();
  if(s.length()==0) return;

  if(s=="PING"){ Serial.println("PONG"); return; }
  if(s=="STATUS"){ printStatus(); return; }

  // ✅ Buzzer command (allowed even during shutdown)
  //   BEEP                -> 120ms
  //   BEEP <ms>           -> duration
  //   BEEP <ms> <freq>    -> duration + freq (มีประโยชน์ถ้าเป็น passive)
  if (s.startsWith("BEEP")) {
    int dur = 120;
    int freq = 2000;
    int n = sscanf(s.c_str(), "BEEP %d %d", &dur, &freq);
    if (n >= 1) dur = (int)clampf(dur, 10, 2000);
    if (n >= 2) freq = (int)clampf(freq, 50, 8000);

    buzzerStart_ms(dur, freq);
    Serial.println("OK");
    return;
  }

  // ระหว่าง shutdown ไม่รับคำสั่ง motion ใหม่
  if(shutdownPending){
    Serial.println("ERR_SHUTTING_DOWN");
    return;
  }

  if (s.startsWith("RGB ")) {
    int r,g,b;
    if(sscanf(s.c_str(),"RGB %d %d %d",&r,&g,&b)==3){
      ledMode = MODE_MANUAL; ledDirty=false;
      strip.setBrightness(brightnessVal);
      setAll((uint8_t)clampf(r,0,255),(uint8_t)clampf(g,0,255),(uint8_t)clampf(b,0,255));
      Serial.println("OK");
    } else Serial.println("ERR");
    return;
  }

  if (s.startsWith("LIGHT ")) {
    String m=s.substring(6); m.trim(); m.toUpperCase();
    if(m=="WHITE") setLedMode(MODE_WHITE);
    else if(m=="RED") setLedMode(MODE_RED);
    else if(m=="SUNRISE") setLedMode(MODE_SUNRISE);
    else if(m=="BREATH") setLedMode(MODE_BREATH);
    else if(m=="RAINBOW") setLedMode(MODE_RAINBOW);
    else if(m=="POLICE") setLedMode(MODE_POLICE);
    else if(m=="OFF") setLedMode(MODE_OFF);
    else { Serial.println("ERR"); return; }
    Serial.println("OK"); return;
  }

  if (s.startsWith("BR ")) {
    brightnessVal=(uint8_t)clampf(s.substring(3).toFloat(),0,255);
    ledDirty=true;
    Serial.println("OK");
    return;
  }

  if (!armed) {
    Serial.println("ERR_NOT_ARMED");
    return;
  }

  if (s=="HOLD ALL") { holdAll=true; scanOn=false; Serial.println("OK"); return; }
  if (s=="STOPSCAN") { scanOn=false; Serial.println("OK"); return; }

  if (s=="HOME") {
    scanOn=false;
    bookHold=false;
    for(int i=0;i<N;i++) tgtA[i]=0;
    holdAll=false;
    Serial.println("OK");
    return;
  }

  if (s.startsWith("BOOK_HOLD ")) {
    String v=s.substring(10); v.trim(); v.toUpperCase();
    if(v=="ON"){
      for(int i=1;i<N;i++) bookLockPose[i-1]=curA[i];
      bookHold=true; scanOn=false; holdAll=false;
      Serial.println("OK");
    } else if(v=="OFF"){
      bookHold=false;
      Serial.println("OK");
    } else Serial.println("ERR");
    return;
  }

  if (s.startsWith("SCAN ")) {
    if (bookHold) { Serial.println("ERR_LOCKED"); return; }

    char w[8]; int mn,mx,spd;
    if(sscanf(s.c_str(),"SCAN %7s %d %d %d",w,&mn,&mx,&spd)==4 && String(w)=="S1"){
      scanMin = clampf(mn,0,180);
      scanMax = clampf(mx,0,180);
      if(scanMin>scanMax){ float t=scanMin; scanMin=scanMax; scanMax=t; }

      scanSpeedDegPerSec = clampf(spd,1,180);
      scanDir = 1;
      scanOn = true;
      holdAll = false;

      setLedMode(MODE_RED);
      Serial.println("OK");
    } else Serial.println("ERR");
    return;
  }

  if (s.startsWith("POSE ")) {
    if (bookHold) { Serial.println("ERR_LOCKED"); return; }

    int a[5]; int tms=1200;
    int n = sscanf(s.c_str(), "POSE %d %d %d %d %d %d", &a[0],&a[1],&a[2],&a[3],&a[4],&tms);
    if(n>=5){
      for(int i=0;i<5;i++) tgtA[i]=clampf(a[i],0,180);

      if(n==6){
        float maxDiff=0;
        for(int i=0;i<N;i++){
          float d=fabsf(tgtA[i]-curA[i]);
          if(d>maxDiff) maxDiff=d;
        }
        float sec=clampf(tms/1000.0f,0.2f,10.0f);
        moveMaxSpeedDegPerSec=clampf(maxDiff/sec,10.0f,160.0f);
      } else {
        moveMaxSpeedDegPerSec=60.0f;
      }

      holdAll=false;
      scanOn=false;
      Serial.println("OK");
    } else Serial.println("ERR");
    return;
  }

  Serial.println("ERR_UNKNOWN");
}

static bool readPressedRaw() { return (digitalRead(ENABLE_PIN)==LOW); }

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, INPUT_PULLUP);

  // ✅ BUZZER init
  pinMode(BUZZER_PIN, OUTPUT);
  buzzerOffNow();

  strip.begin();
  strip.setBrightness(brightnessVal);
  strip.clear();
  strip.show();

  enterSafeNow();
  lastLedTick = millis();
  lastMoveTick = millis();
  Serial.println("READY");
}

void loop() {
  // ✅ buzzer non-blocking update
  buzzerUpdate();

  // ปุ่ม: toggle
  bool raw = readPressedRaw();
  if(raw != lastRaw){ lastDebounceMs = millis(); lastRaw = raw; }
  if((millis()-lastDebounceMs) > debounceMs){
    if(raw != stable){
      stable = raw;
      if(stable){
        if(!armed){
          enterArmedNow();
        } else {
          if(!shutdownPending) beginShutdown();
        }
      }
    }
  }

  // Serial commands
  while(Serial.available()){
    char c=(char)Serial.read();
    if(c=='\n'){
      line.trim();
      if(line.length()>0) handleCommand(line);
      line="";
    } else if(c!='\r'){
      line += c;
      if(line.length()>180) line="";
    }
  }

  applyLedMode();
  updateMotion();

  if(shutdownPending && nearAll(0.0f, 0.8f)){
    finishShutdown();
  }
}
