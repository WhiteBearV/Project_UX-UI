#!/usr/bin/env python3
import time
import cv2
import lgpio
from picamera2 import Picamera2

# ========= CONFIG =========
SERVO = [
    {"pin": 12, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S1 base pan
    {"pin": 16, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 90.0,  "max_accel": 220.0},  # S2 reach (in/out)
    {"pin": 26, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S3 arm up/down
    {"pin": 20, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S4 lamp up/down
    {"pin": 21, "min_us": 650, "max_us": 2350, "range_deg": 180.0, "max_speed": 120.0, "max_accel": 260.0},  # S5 lamp pan
]

# ทิศทาง: ถ้าหมุนผิดทาง ให้กลับเครื่องหมายเฉพาะตัวนั้น (+1/-1)
# (คุณบอกว่า "กลับด้านทุกตัว" เลยตั้ง -1 ทั้งหมดไว้ก่อน)
DIR_S1 = -1
DIR_S2 = -1
DIR_S3 = -1
DIR_S4 = -1
DIR_S5 = -1

FRAME_W, FRAME_H = 1600, 900

# ถ้า SSH ไม่มีจอ ให้ False (ไม่ใช้ cv2.imshow)
SHOW = True

# เฟรม/การควบคุม
SERVO_UPDATE_SEC = 0.02      # 50Hz สำหรับ "ขยับไปยัง target"
DETECT_INTERVAL_SEC = 5.0    # ตรวจจับทุก 5 วินาที (ลดกระตุกตามที่ต้องการ)

# เงื่อนไข "อยู่นิ่ง"
DEADBAND_X = 0.03
DEADBAND_Y = 0.03
DEADBAND_Z = 0.01
TARGET_CHANGE_EPS_DEG = 1.0  # ถ้า target เปลี่ยนน้อยกว่า 1° ถือว่านิ่ง
AT_TARGET_EPS_DEG = 0.5      # ถึงเป้าถ้าคลาดไม่เกินนี้

# เป้าหมายขนาดคนในภาพ (แทนระยะ)
SIZE_SP = 0.18

# DNN model
PROTO = "opencv_face_detector.prototxt"
MODEL = "opencv_face_detector.caffemodel"
CONF_TH = 0.5
PERSON_CLASS_ID = 15  # VOC person

# ========= Utils =========
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def angle_to_us(cfg, angle):
    rng = float(cfg["range_deg"])
    a = clamp(float(angle), 0.0, rng)
    return int(cfg["min_us"] + (a / rng) * (cfg["max_us"] - cfg["min_us"]))

# ========= PID (output = deg/s) =========
class PID:
    def __init__(self, kp, ki, kd, out_limit, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = float(out_limit)  # deg/s
        self.i_limit = float(i_limit)
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def update(self, e, t_now):
        if self.prev_t is None:
            self.prev_t = t_now
            self.prev_e = e
            return 0.0
        dt = t_now - self.prev_t
        if dt <= 1e-4:
            return 0.0

        # เราตรวจจับห่าง (เช่น 5s) ถ้าใช้ dt ตรงๆ จะ “กระชาก” ได้
        # clamp dt เพื่อลด derivative/integral spike (คุมให้นิ่ม)
        dt = min(dt, 0.25)

        self.i += e * dt
        self.i = clamp(self.i, -self.i_limit, self.i_limit)

        de = (e - self.prev_e) / dt
        out = self.kp * e + self.ki * self.i + self.kd * de
        out = clamp(out, -self.out_limit, self.out_limit)

        self.prev_e = e
        self.prev_t = t_now
        return out

# ========= Servo smoothing (speed+accel limited) =========
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

def all_at_target(current_deg, target_deg, eps=AT_TARGET_EPS_DEG):
    return all(abs(current_deg[i] - target_deg[i]) <= eps for i in range(5))

def main():
    # --- GPIO setup ---
    chip = lgpio.gpiochip_open(0)
    pins = [c["pin"] for c in SERVO]
    for p in pins:
        try:
            lgpio.gpio_claim_output(chip, p)
        except Exception:
            pass

    # --- Servo state ---
    n = 5
    current_deg = [0.0] * n
    target_deg  = [0.0] * n
    vel_deg_s   = [0.0] * n

    # เริ่มต้น: ไปที่ 0° แล้ว "อยู่นิ่ง"
    for i in range(n):
        lgpio.tx_servo(chip, pins[i], angle_to_us(SERVO[i], 0.0))

    # --- Camera ---
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

    # --- DNN ---
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

    # --- PID tuning ---
    pid_x_base = PID(55.0, 0.0, 10.0, out_limit=120.0, i_limit=0.25)  # S1
    pid_y_arm  = PID(45.0, 0.0, 9.0,  out_limit=90.0,  i_limit=0.25)  # S3
    pid_z_reach= PID(70.0, 0.0, 12.0, out_limit=60.0,  i_limit=0.20)  # S2
    pid_x_lamp = PID(30.0, 0.0, 7.0,  out_limit=60.0,  i_limit=0.15)  # S5
    pid_y_lamp = PID(30.0, 0.0, 7.0,  out_limit=60.0,  i_limit=0.15)  # S4

    cx0, cy0 = FRAME_W / 2.0, FRAME_H / 2.0

    last_bbox = None
    last_det_t = 0.0
    last_servo_update = time.time()

    # โหมดการทำงาน
    state = "IDLE"  # IDLE: อยู่ที่ 0° รอคน, TRACK: มี target แล้วกำลังขยับ/นิ่ง

    # ใช้สำหรับ overlay
    cx, cy = cx0, cy0

    def go_home_and_wait(timeout_sec=8.0):
        nonlocal last_servo_update, state
        for k in range(5):
            target_deg[k] = 0.0
        state = "IDLE"
        t0 = time.time()
        while True:
            now2 = time.time()
            if (now2 - last_servo_update) >= SERVO_UPDATE_SEC:
                dt2 = now2 - last_servo_update
                last_servo_update = now2
                for k in range(5):
                    step_positional_smooth(
                        k, dt2, SERVO[k], pins,
                        current_deg, target_deg, vel_deg_s, chip
                    )
                if all(abs(current_deg[k] - 0.0) < AT_TARGET_EPS_DEG for k in range(5)):
                    break
            if (now2 - t0) > timeout_sec:
                break
            time.sleep(0.002)

    try:
        while True:
            now = time.time()

            # ----- capture frame -----
            rgb = cam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ----- key handling (Q กลับบ้าน) -----
            if SHOW:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    go_home_and_wait(timeout_sec=8.0)
                    break

            # ----- detection: ทำเป็นช่วง (ทุก 5 วินาที) -----
            detected_now = False
            if (now - last_det_t) >= DETECT_INTERVAL_SEC:
                last_det_t = now

                blob = cv2.dnn.blobFromImage(
                    bgr, scalefactor=1.0, size=(300, 300),
                    mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
                )
                net.setInput(blob)
                det = net.forward()

                best = None
                best_conf = 0.0
                for i in range(det.shape[2]):
                    conf = float(det[0, 0, i, 2])
                    if conf < CONF_TH:
                        continue

                    x1 = int(det[0, 0, i, 3] * FRAME_W)
                    y1 = int(det[0, 0, i, 4] * FRAME_H)
                    x2 = int(det[0, 0, i, 5] * FRAME_W)
                    y2 = int(det[0, 0, i, 6] * FRAME_H)

                    x1 = clamp(x1, 0, FRAME_W - 1)
                    y1 = clamp(y1, 0, FRAME_H - 1)
                    x2 = clamp(x2, 0, FRAME_W - 1)
                    y2 = clamp(y2, 0, FRAME_H - 1)

                    if conf > best_conf and x2 > x1 and y2 > y1:
                        best_conf = conf
                        best = (x1, y1, x2, y2, conf)

                if best is not None:
                    last_bbox = best
                    detected_now = True
                    state = "TRACK"
                else:
                    last_bbox = None
                    detected_now = False
                    # ถ้าไม่เจอคน ให้กลับไปนิ่งที่ 0° และ reset PID
                    state = "IDLE"
                    pid_x_base.reset(); pid_y_arm.reset(); pid_z_reach.reset(); pid_x_lamp.reset(); pid_y_lamp.reset()
                    for k in range(5):
                        target_deg[k] = 0.0

            # ----- ถ้าเพิ่งเจอคน (หรือมี bbox ล่าสุด) ให้คำนวณ target "ครั้งเดียว" แล้วค่อยขยับไป -----
            if state == "TRACK" and last_bbox is not None and detected_now:
                x1, y1, x2, y2, conf = last_bbox
                bw = (x2 - x1)
                bh = (y2 - y1)
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0

                ex = (cx - cx0) / cx0
                ey = (cy - cy0) / cy0

                if abs(ex) < DEADBAND_X: ex = 0.0
                if abs(ey) < DEADBAND_Y: ey = 0.0

                size = (bh / FRAME_H)
                ez = (SIZE_SP - size)
                if abs(ez) < DEADBAND_Z: ez = 0.0

                # PID outputs = deg/s แต่เรา "คำนวณเป็นก้าว" แล้วค่อยขยับไป (ลดกระตุก)
                # dt_cmd = ขนาดก้าวของคำสั่ง (ยิ่งเล็กยิ่งนิ่ง)
                dt_cmd = 0.25

                vx_base = pid_x_base.update(ex, now) * DIR_S1
                vy_arm  = pid_y_arm.update(ey, now)  * DIR_S3
                vz_reach= pid_z_reach.update(ez, now)* DIR_S2
                vx_lamp = pid_x_lamp.update(ex, now) * DIR_S5
                vy_lamp = pid_y_lamp.update(ey, now) * DIR_S4

                new_targets = [
                    clamp(target_deg[0] + vx_base * dt_cmd, 0.0, 180.0),  # S1
                    clamp(target_deg[1] + vz_reach * dt_cmd, 0.0, 180.0), # S2
                    clamp(target_deg[2] + vy_arm  * dt_cmd, 0.0, 180.0),  # S3
                    clamp(target_deg[3] + vy_lamp * dt_cmd, 0.0, 180.0),  # S4
                    clamp(target_deg[4] + vx_lamp * dt_cmd, 0.0, 180.0),  # S5
                ]

                # ถ้าเปลี่ยน target น้อยมาก ให้ "อยู่นิ่ง"
                if all(abs(new_targets[k] - target_deg[k]) < TARGET_CHANGE_EPS_DEG for k in range(5)):
                    # ไม่อัปเดต target -> คงเดิม
                    pass
                else:
                    for k in range(5):
                        target_deg[k] = new_targets[k]

            # ----- servo update: ขยับไปหา target เท่านั้น -----
            if (now - last_servo_update) >= SERVO_UPDATE_SEC:
                dt = now - last_servo_update
                last_servo_update = now

                # ถ้า IDLE และอยู่ที่ 0 แล้ว ก็ไม่ต้องคำนวณเยอะ (ลดกระตุก)
                if state == "IDLE" and all(abs(current_deg[k] - 0.0) <= AT_TARGET_EPS_DEG for k in range(5)):
                    # ค้างไว้เฉยๆ
                    pass
                else:
                    for k in range(5):
                        step_positional_smooth(
                            k, dt, SERVO[k], pins,
                            current_deg, target_deg, vel_deg_s, chip
                        )

            # ----- preview -----
            if SHOW:
                # จุดโฟกัส (เขียว)
                cv2.circle(bgr, (int(cx0), int(cy0)), 10, (0, 255, 0), -1)
                cv2.circle(bgr, (int(cx0), int(cy0)), 20, (0, 255, 0), 2)

                if last_bbox is not None:
                    x1, y1, x2, y2, conf = last_bbox
                    # กรอบแดง
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # จุดกลางของคน (น้ำเงิน)
                    cv2.circle(bgr, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                    cv2.putText(
                        bgr, f"person {conf:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )

                cv2.putText(
                    bgr,
                    f"STATE: {state}  detect_every={DETECT_INTERVAL_SEC:.0f}s",
                    (10, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("track", bgr)

            time.sleep(0.001)

    finally:
        # พอกำลังจะออก ให้กลับ 0° ก่อน (ถ้าอยากให้แน่นอนที่สุด)
        try:
            for k in range(5):
                target_deg[k] = 0.0
            t0 = time.time()
            last_t = time.time()
            while time.time() - t0 < 2.5:
                now2 = time.time()
                dt2 = now2 - last_t
                last_t = now2
                for k in range(5):
                    step_positional_smooth(k, dt2, SERVO[k], pins, current_deg, target_deg, vel_deg_s, chip)
                if all(abs(current_deg[k] - 0.0) <= AT_TARGET_EPS_DEG for k in range(5)):
                    break
                time.sleep(0.01)
        except Exception:
            pass

        for p in pins:
            try:
                lgpio.tx_servo(chip, p, 0)
            except Exception:
                pass

        try:
            lgpio.gpiochip_close(chip)
        except Exception:
            pass

        try:
            cam.stop()
        except Exception:
            pass

        if SHOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
