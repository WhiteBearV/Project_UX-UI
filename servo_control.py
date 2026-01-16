import time
import json
import os
import lgpio
import curses

# ========= CONFIG =========
SERVO_PINS = [12, 16, 26]   # <-- แก้ตามที่ต่อจริง (Servo1, Servo2, Servo3)

MIN_US = 650
MAX_US = 2350

STEP_US = 10
SERVO_UPDATE_SEC = 0.02      # 50Hz

ANGLE_STEP_INIT = 2.0
MIN_ANGLE_STEP = 0.5
MAX_ANGLE_STEP = 15.0

SAVE_FILE = "pose_3servos.json"

DETACH_ON_IDLE = True
IDLE_SECONDS = 0.35

HOLD_BETWEEN_KEYS = 0.15
# ==========================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def angle_to_us(a):
    a = clamp(float(a), 0.0, 180.0)
    return int(MIN_US + (a / 180.0) * (MAX_US - MIN_US))

def us_to_angle(us):
    us = clamp(float(us), MIN_US, MAX_US)
    return (us - MIN_US) * 180.0 / (MAX_US - MIN_US)

def stop_pulses_safe(h, gpio):
    # หยุดพัลส์แบบไม่แครช (บางเครื่อง tx_servo(...,0) ไม่รับ)
    try:
        lgpio.tx_servo(h, gpio, 0)
        return True
    except Exception:
        pass
    for duty in (50, 0):
        try:
            lgpio.tx_pwm(h, gpio, 0, duty)
            return True
        except Exception:
            pass
    return False

def load_poses():
    if not os.path.exists(SAVE_FILE):
        return []
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_poses(poses):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(poses, f, ensure_ascii=False, indent=2)

def run(stdscr):
    # ---- curses setup ----
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()

    # ---- gpio setup ----
    h = lgpio.gpiochip_open(0)

    for pin in SERVO_PINS:
        try:
            lgpio.gpio_claim_output(h, pin)
        except Exception:
            pass

    def set_servo_us(pin, us):
        lgpio.tx_servo(h, pin, int(us))

    poses = load_poses()

    n = len(SERVO_PINS)
    selected = 0  # 0..2 แสดง servo 1..3
    angle_step = ANGLE_STEP_INIT

    current_us = [angle_to_us(0.0)] * n
    target_us  = [angle_to_us(0.0)] * n
    pulses_active = [True] * n

    for i, pin in enumerate(SERVO_PINS):
        set_servo_us(pin, current_us[i])

    last_input_time = time.time()
    last_servo_update = time.time()

    # ---- log เฉพาะ l/p ----
    log_lines = []

    def userlog(msg):
        log_lines.append(msg)
        if len(log_lines) > 400:
            del log_lines[:150]

    def draw(status_msg=""):
        stdscr.erase()

        now_angles = [us_to_angle(u) for u in current_us]
        tgt_angles = [us_to_angle(u) for u in target_us]
        saved = len(poses)

        # ทำเป็น list ของบรรทัด (ห้ามใส่ \n ใน addstr)
        lines = [
            f"Sel: {selected+1}    saved: {saved}    step: {angle_step:.1f}°",
            f"S1  {now_angles[0]:6.1f} -> {tgt_angles[0]:6.1f}",
            f"S2  {now_angles[1]:6.1f} -> {tgt_angles[1]:6.1f}",
            f"S3  {now_angles[2]:6.1f} -> {tgt_angles[2]:6.1f}",
            "1-3 select | <-/+  ->/- | UP/DN step | s=save l=list p=play h=home c=clear q=quit",
        ]

        if status_msg:
            lines.append(status_msg)

        # วาดทีละบรรทัด พร้อมตัดให้พอดีความกว้างหน้าจอ
        max_cols = max(1, curses.COLS - 1)
        max_rows = curses.LINES

        row = 0
        for line in lines:
            if row >= max_rows:
                break
            stdscr.addstr(row, 0, line[:max_cols])
            row += 1

        # log area (ถ้าคุณมี)
        start_row = row + 1
        max_log_rows = max_rows - start_row
        if max_log_rows > 0:
            tail = log_lines[-max_log_rows:]
            for i, line in enumerate(tail):
                stdscr.addstr(start_row + i, 0, line[:max_cols])

        stdscr.refresh()


    def ensure_pulses_on(i):
        if not pulses_active[i]:
            set_servo_us(SERVO_PINS[i], current_us[i])
            pulses_active[i] = True

    def set_target_angle(i, angle):
        target_us[i] = angle_to_us(angle)

    def step_servo_toward_target(i):
        if current_us[i] == target_us[i]:
            return
        if abs(current_us[i] - target_us[i]) <= STEP_US:
            current_us[i] = target_us[i]
        else:
            current_us[i] += STEP_US if current_us[i] < target_us[i] else -STEP_US
        set_servo_us(SERVO_PINS[i], current_us[i])

    def all_reached():
        return all(current_us[i] == target_us[i] for i in range(n))

    def idle_stop_if_needed():
        if not DETACH_ON_IDLE:
            return
        if not all_reached():
            return
        if (time.time() - last_input_time) <= IDLE_SECONDS:
            return

        # หยุดพัลส์ทุกตัว
        for i, pin in enumerate(SERVO_PINS):
            if pulses_active[i]:
                ok = stop_pulses_safe(h, pin)
                pulses_active[i] = False if ok else True

    def wait_until_reached(timeout_sec=6.0):
        nonlocal last_servo_update
        start = time.time()
        while not all_reached() and (time.time() - start) < timeout_sec:
            now = time.time()
            if (now - last_servo_update) >= SERVO_UPDATE_SEC:
                for i in range(n):
                    if pulses_active[i]:
                        step_servo_toward_target(i)
                last_servo_update = now
            time.sleep(0.002)

    def save_pose():
        pose = [round(us_to_angle(u), 2) for u in target_us]
        poses.append({"t": round(time.time(), 3), "angles": pose})
        save_poses(poses)
        userlog(f"saved pose {len(poses):02d}: {pose}")

    def list_poses():
        if not poses:
            userlog("(no poses)")
            return
        userlog("=== Poses ===")
        for i, e in enumerate(poses, 1):
            ang = e.get("angles", [0, 0, 0])
            userlog(f"{i:02d}) {ang}")
        userlog("=== end ===")

    def playback():
        if not poses:
            userlog("No poses to play.")
            return
        userlog(f"=== Playback ({len(poses)} poses) ===")

        for k, e in enumerate(poses, 1):
            ang = e.get("angles", [0, 0, 0])
            # เปิดพัลส์ทุกตัว
            for i in range(n):
                ensure_pulses_on(i)
                set_target_angle(i, clamp(float(ang[i]), 0.0, 180.0))

            userlog(f"[{k:02d}/{len(poses):02d}] -> {ang}")
            draw("PLAYING...")
            wait_until_reached(timeout_sec=8.0)
            time.sleep(HOLD_BETWEEN_KEYS)

        userlog("=== Playback done ===")

    # initial draw
    draw()

    try:
        while True:
            key = stdscr.getch()
            if key != -1:
                last_input_time = time.time()

                # เลือกเซอร์โวด้วยเลข 1-3
                if key in (ord('1'), ord('2'), ord('3')):
                    selected = key - ord('1')
                    selected = clamp(selected, 0, n-1)

                elif key == ord('q'):
                    break

                elif key == curses.KEY_LEFT:   # เพิ่มองศา
                    ensure_pulses_on(selected)
                    a = us_to_angle(target_us[selected]) + angle_step
                    set_target_angle(selected, clamp(a, 0.0, 180.0))

                elif key == curses.KEY_RIGHT:  # ลดองศา
                    ensure_pulses_on(selected)
                    a = us_to_angle(target_us[selected]) - angle_step
                    set_target_angle(selected, clamp(a, 0.0, 180.0))

                elif key == curses.KEY_UP:
                    angle_step = clamp(angle_step + 0.5, MIN_ANGLE_STEP, MAX_ANGLE_STEP)

                elif key == curses.KEY_DOWN:
                    angle_step = clamp(angle_step - 0.5, MIN_ANGLE_STEP, MAX_ANGLE_STEP)

                elif key == ord('h'):
                    # home เฉพาะตัวที่เลือก
                    ensure_pulses_on(selected)
                    set_target_angle(selected, 0.0)

                elif key == ord('s'):
                    save_pose()

                elif key == ord('c'):
                    poses.clear()
                    save_poses(poses)
                    userlog("Cleared all poses.")

                elif key == ord('l'):
                    list_poses()

                elif key == ord('p'):
                    playback()

            # servo update @50Hz
            now = time.time()
            if (now - last_servo_update) >= SERVO_UPDATE_SEC:
                for i in range(n):
                    if pulses_active[i]:
                        step_servo_toward_target(i)
                last_servo_update = now

            # idle stop (ลดสั่นตอนนิ่ง)
            idle_stop_if_needed()

            draw()
            time.sleep(0.002)

    finally:
        # ปิดพัลส์ทุกตัวแบบปลอดภัย
        for pin in SERVO_PINS:
            try:
                stop_pulses_safe(h, pin)
            except Exception:
                pass
        lgpio.gpiochip_close(h)

if __name__ == "__main__":
    curses.wrapper(run)
