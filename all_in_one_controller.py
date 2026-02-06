#!/usr/bin/env python3
# all_in_one_controller.py
# - YOLO: person+book
# - Servo: 5 servos (lgpio) + smoothing
# - LED: ส่งคำสั่งผ่าน Serial ไป Arduino (สเก็ตช์ LedRgb.ino)

import time
import random
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import serial
import lgpio

# ================== CONFIG ==================
# --- Camera / YOLO ---
FRAME_W, FRAME_H = 1280, 720
SHOW = True   # ถ้า SSH ไม่มีจอ ให้ False
YOLO_MODEL = "yolo11n.pt"
YOLO_CONF = 0.25

# แยก threshold รายคลาส เพื่อ "แท็กหนังสือง่ายขึ้น" และ "แท็กคนน้อยลง"
PERSON_CONF = 0.55   # เพิ่มค่า = เจอคนน้อยลง/มั่วลดลง
BOOK_CONF   = 0.10   # ลดค่า = เจอหนังสือง่ายขึ้น
PERSON_CONFIRM_FRAMES = 3   # ต้องเจอคนติดกันกี่เฟรมถึงถือว่าเจอจริง
PERSON_MIN_AREA_FRAC  = 0.02  # กล่องคนเล็กกว่า % นี้ของภาพ -> ตัดทิ้ง (ลด false positive)
BOOK_MIN_AREA_FRAC    = 0.002 # กันกล่องหนังสือเล็กจิ๋วที่มั่ว

# COCO class ids
CLS_PERSON = 0
CLS_BOOK = 73

# --- LED Serial (Arduino) ---
SERIAL_PORT = "/dev/ttyACM0"  # ปรับตามของคุณ
SERIAL_BAUD = 115200

# --- Logic timers ---
BOOK_HOLD_SEC = 3.0
WHITE_ON_SEC = 60.0

# --- 5 Servo config ---
SERVO = [
    {"pin": 12, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S1 base pan
    {"pin": 16, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed":  90.0, "max_accel": 220.0},  # S2 reach
    {"pin": 26, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S3 arm
    {"pin": 20, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S4 lamp tilt
    {"pin": 21, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S5 lamp pan
]
# ถ้าหมุนผิดทาง ให้สลับเครื่องหมายของตัวนั้น (+1/-1)
DIR = [-1, -1, -1, -1, -1]

# ท่าทาง 5 ท่า (ปรับองศาได้ตามแขนคุณ)
# รูปแบบ: [S1_base, S2_reach, S3_arm, S4_lamp_tilt, S5_lamp_pan]
# *ตามโจทย์ตอนนี้ "โคมไม่ต้องขยับตามคน" -> เราจะล็อก S4,S5 ไว้ที่ 0 ตลอด
POSES_DEG = [
    [ 90,  20,  30,   0,   0],  # pose 1
    [ 70,  40,  60,   0,   0],  # pose 2
    [110,  35,  80,   0,   0],  # pose 3
    [ 60,  15, 110,   0,   0],  # pose 4
    [120,  55,  40,   0,   0],  # pose 5
]

SERVO_UPDATE_SEC = 0.02  # 50Hz
DEADBAND_X = 0.03
DEADBAND_Y = 0.03
TARGET_CHANGE_EPS_DEG = 1.0
AT_TARGET_EPS_DEG = 0.5

# --- Servo behavior (ลดสั่นด้วย state-based motion) ---
STARTUP_FREEZE_SEC = 10.0   # เริ่มโปรแกรม: ไม่ขยับอะไรเลย
SEARCH_SWEEP_MIN_DEG = 20.0 # ช่วงกวาดฐานเพื่อหาคน
SEARCH_SWEEP_MAX_DEG = 160.0
SEARCH_SWEEP_SPEED_DPS = 20.0  # deg/s ความเร็วหมุนฐานตอนหา
POSE_MIN_HOLD_SEC = 4.0     # อยู่ท่าอย่างน้อยกี่วิ
POSE_MAX_HOLD_SEC = 8.0
POSE_TRANSITION_MAX_SPEED = 70.0  # จำกัดความเร็วเซอร์โวตอนเปลี่ยนท่า (ลดกระชาก/สั่น)

# ================== Utils ==================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def angle_to_us(cfg, angle):
    rng = float(cfg["range_deg"])
    a = clamp(float(angle), 0.0, rng)
    return int(cfg["min_us"] + (a / rng) * (cfg["max_us"] - cfg["min_us"]))


def step_positional_smooth(i, dt, cfg, pins, current_deg, target_deg, vel_deg_s, chip):
    err = target_deg[i] - current_deg[i]
    if abs(err) < 0.2:
        current_deg[i] = target_deg[i]
        vel_deg_s[i] = 0.0
        lgpio.tx_servo(chip, pins[i], angle_to_us(cfg, current_deg[i]))
        return

    max_speed = float(cfg.get("max_speed", 120.0))
    max_accel = float(cfg.get("max_accel", 260.0))

    desired_v = clamp(err / max(dt, 1e-6), -max_speed, max_speed)
    dv = desired_v - vel_deg_s[i]
    max_dv = max_accel * dt
    dv = clamp(dv, -max_dv, max_dv)
    vel_deg_s[i] += dv

    nxt = current_deg[i] + vel_deg_s[i] * dt

    # กัน overshoot
    if (err > 0 and nxt > target_deg[i]) or (err < 0 and nxt < target_deg[i]):
        nxt = target_deg[i]
        vel_deg_s[i] = 0.0

    current_deg[i] = nxt
    lgpio.tx_servo(chip, pins[i], angle_to_us(cfg, current_deg[i]))

# ================== LED Serial ==================
class LedSerial:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # รอ Arduino reset หลังเปิดพอร์ต
        self._drain()

    def _drain(self):
        try:
            while self.ser.in_waiting:
                self.ser.readline()
        except Exception:
            pass

    def send(self, s: str):
        self.ser.write((s + "\n").encode())
        self.ser.flush()
        time.sleep(0.01)
        # ไม่บังคับต้องอ่านตอบกลับ แต่เก็บไว้ debug ได้
        while self.ser.in_waiting:
            _ = self.ser.readline()

    def clear(self):
        self.send("CLEAR")

    def white(self, br=120):
        self.send(f"BR {int(br)}")
        self.send("FILL 255 255 255")

    def rainbow_step(self, delay_ms=5):
        self.send(f"RAINBOW {int(delay_ms)}")
        # ถ้าอยากให้ “ไล่” ชัดขึ้น เปิดใช้ rotate ได้:
        # self.send("ROTATE 1")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

# ================== Main ==================
def main():
    # --- Servo init ---
    chip = lgpio.gpiochip_open(0)
    pins = [c["pin"] for c in SERVO]
    for p in pins:
        try:
            lgpio.gpio_claim_output(chip, p)
        except Exception:
            pass

    n = 5
    current_deg = [0.0] * n
    target_deg = [0.0] * n
    vel_deg_s = [0.0] * n
    for i in range(n):
        lgpio.tx_servo(chip, pins[i], angle_to_us(SERVO[i], 0.0))

    # --- Camera ---
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

    # --- YOLO ---
    model = YOLO(YOLO_MODEL)

    # --- LED Serial ---
    led = LedSerial(SERIAL_PORT, SERIAL_BAUD)
    led.clear()

    # --- State machine ---
    mode = "IDLE"        # IDLE | PERSON | WHITE
    person_hit = 0  # นับเฟรมที่เจอคนติดกัน
    book_t0 = None
    white_until = 0.0

    # --- Servo state machine ---
    t_start = time.time()
    servo_mode = "FREEZE"   # FREEZE -> SEARCH -> POSE
    sweep_dir = 1
    sweep_angle = 90.0
    current_pose = None
    next_pose_change = 0.0

    last_servo_update = time.time()
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            now = time.time()

            rgb = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # YOLO detect (person+book)
            res = model(frame, classes=[CLS_PERSON, CLS_BOOK], conf=YOLO_CONF, verbose=False)[0]

            best_person = None  # (conf, x1,y1,x2,y2, area)
            best_book = None    # (conf, x1,y1,x2,y2, area)

            frame_area = float(FRAME_W * FRAME_H)

            for b in res.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = max(0, x2 - x1) * max(0, y2 - y1)

                if cls == CLS_PERSON:
                    # คน: เข้มขึ้น (conf สูง + กล่องต้องใหญ่พอ)
                    if conf >= PERSON_CONF and (area / frame_area) >= PERSON_MIN_AREA_FRAC:
                        if best_person is None or conf > best_person[0]:
                            best_person = (conf, x1, y1, x2, y2, area)

                elif cls == CLS_BOOK:
                    # หนังสือ: ผ่อนลง (conf ต่ำกว่า + กันกล่องเล็กจิ๋ว)
                    if conf >= BOOK_CONF and (area / frame_area) >= BOOK_MIN_AREA_FRAC:
                        if best_book is None or conf > best_book[0]:
                            best_book = (conf, x1, y1, x2, y2, area)

                if SHOW:
                    label = "person" if cls == CLS_PERSON else "book"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            person_raw = best_person is not None
            book_raw = best_book is not None

            # คอนเฟิร์ม "คน" ด้วยจำนวนเฟรม (ลด false positive)
            if person_raw:
                person_hit = min(PERSON_CONFIRM_FRAMES, person_hit + 1)
            else:
                person_hit = max(0, person_hit - 1)

            person_seen = (person_hit >= PERSON_CONFIRM_FRAMES)
            book_seen = book_raw
            # ---------------- LED Logic ----------------
            # เงื่อนไขใหม่:
            # 1) ถ้าเจอหนังสือ หรือเจอหนังสือ+คน "ค้างครบ 3 วินาที" -> เข้า WHITE 1 นาที
            # 2) ระหว่าง WHITE: ค้างไฟขาวให้ครบ 1 นาที (ไม่ดับทันที)
            # 3) ครบ 1 นาที: ถ้ายังเจอคน -> ดับไฟ (CLEAR) แล้วกลับ IDLE

            if mode == "WHITE":
                if now >= white_until:
                    # ครบ 1 นาทีแล้ว -> เช็คคน ณ ตอนนี้
                    if person_seen:
                        led.clear()  # ดับไฟตามโจทย์
                    mode = "IDLE"
                    book_t0 = None
                else:
                    # ค้างไฟขาวไว้ ไม่ยิงซ้ำ
                    pass

            else:
                # ไม่ได้อยู่ใน WHITE -> นับเวลาหนังสือ (ไม่สนว่ามีคนหรือไม่)
                if book_seen:
                    if book_t0 is None:
                        book_t0 = now
                    elif (now - book_t0) >= BOOK_HOLD_SEC:
                        mode = "WHITE"
                        white_until = now + WHITE_ON_SEC
                        led.white(br=120)
                        book_t0 = None
                else:
                    book_t0 = None

                # ถ้าไม่ได้เข้า WHITE ให้ทำเอฟเฟกต์คน
                if mode != "WHITE":
                    if person_seen:
                        mode = "PERSON"
                        led.rainbow_step(delay_ms=5)
                    else:
                        mode = "IDLE"
                        led.clear()
            # ---------------- Servo Control (ใหม่ ลดสั่น) ----------------
            # โหมดการขยับ:
            # - FREEZE (0-10s): ไม่ขยับทุกตัว
            # - SEARCH: หลัง 10s หมุนเฉพาะฐาน (S1) เพื่อหาคน
            # - POSE: ถ้าเจอคน -> สุ่มท่าทาง 5 ท่า (S1-S3) / ล็อกโคม (S4,S5)

            # อัปเดต servo_mode
            if (now - t_start) < STARTUP_FREEZE_SEC:
                servo_mode = "FREEZE"
            else:
                if person_seen:
                    if servo_mode != "POSE":
                        servo_mode = "POSE"
                        current_pose = None
                        next_pose_change = 0.0
                else:
                    servo_mode = "SEARCH"

            if servo_mode == "FREEZE":
                # ล็อกทุกตัวไว้ที่ตำแหน่งปัจจุบัน (ไม่เปลี่ยน target)
                # ถ้าอยากให้ "ปิดแรงยึด" จริง ๆ ให้เปลี่ยนเป็น lgpio.tx_servo(..., 0) แต่แขนอาจตก
                pass

            elif servo_mode == "SEARCH":
                # หมุนเฉพาะฐาน S1 ไป-กลับเพื่อหาคน
                dt_sweep = now - last_servo_update
                dt_sweep = max(0.0, min(dt_sweep, 0.1))
                sweep_angle += sweep_dir * SEARCH_SWEEP_SPEED_DPS * dt_sweep
                if sweep_angle >= SEARCH_SWEEP_MAX_DEG:
                    sweep_angle = SEARCH_SWEEP_MAX_DEG
                    sweep_dir = -1
                elif sweep_angle <= SEARCH_SWEEP_MIN_DEG:
                    sweep_angle = SEARCH_SWEEP_MIN_DEG
                    sweep_dir = 1

                target_deg[0] = sweep_angle
                target_deg[1] = 0.0
                target_deg[2] = 0.0
                target_deg[3] = 0.0  # โคมล็อก
                target_deg[4] = 0.0  # โคมล็อก

            elif servo_mode == "POSE":
                # สุ่มท่าทางทุกช่วงเวลา (ถ้ายังเจอคนอยู่)
                if (current_pose is None) or (now >= next_pose_change):
                    current_pose = random.choice(POSES_DEG)
                    next_pose_change = now + random.uniform(POSE_MIN_HOLD_SEC, POSE_MAX_HOLD_SEC)

                # ตั้งเป้าท่า: ขยับเฉพาะ S1-S3, ล็อกโคม S4-S5
                target_deg[0] = float(current_pose[0])
                target_deg[1] = float(current_pose[1])
                target_deg[2] = float(current_pose[2])
                target_deg[3] = 0.0
                target_deg[4] = 0.0

            # ---------------- Servo update note ----------------
            # อัปเดต servo ที่ 50Hz
            if (now - last_servo_update) >= SERVO_UPDATE_SEC:
                dt = now - last_servo_update
                last_servo_update = now
                for k in range(5):
                    # ลดการกระชากตอนเปลี่ยนท่า
                    cfgk = dict(SERVO[k])
                    if servo_mode == "POSE":
                        cfgk["max_speed"] = min(float(cfgk.get("max_speed", 120.0)), POSE_TRANSITION_MAX_SPEED)
                    step_positional_smooth(k, dt, cfgk, pins, current_deg, target_deg, vel_deg_s, chip)

            # ---------------- Preview / FPS ----------------
            if SHOW:
                dtf = now - t_prev
                t_prev = now
                if dtf > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dtf)
                cv2.putText(frame, f"{fps:5.1f} FPS  MODE:{mode}  SERVO:{servo_mode}", (10, FRAME_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("ALL-IN-ONE", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break

    finally:
        # cleanup
        led.clear()
        led.close()

        try:
            cam.stop()
        except Exception:
            pass
        if SHOW:
            cv2.destroyAllWindows()

        # servo off
        for p in pins:
            try:
                lgpio.tx_servo(chip, p, 0)
            except Exception:
                pass
        try:
            lgpio.gpiochip_close(chip)
        except Exception:
            pass

if __name__ == "__main__":
    main()
