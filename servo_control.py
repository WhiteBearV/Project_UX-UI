import time
import json
import os
import lgpio
import curses

# ========= CONFIG =========
# S1-S3: เซอร์โวแบบกำหนด "ตำแหน่ง" (0-180°)
# S4-S5: เซอร์โว 360° แบบ "continuous rotation" (คุมความเร็วด้วย PWM รอบ neutral)
#
# หมายเหตุ:
# - ถ้า S4,S5 ของคุณเป็น 360° แบบ "positional" (กำหนดมุมได้จริง) ให้บอกบ๊อบ เพราะโหมดจะต่างกัน
# - ค่าพัลส์ (us) ของแต่ละรุ่นอาจต่างกัน ควรจูนให้เข้ากับฮาร์ดแวร์จริง
SERVO_CONFIG = [
    # ---- Positional servos (0-180°) ----
    {"pin": 12, "mode": "positional", "range_deg": 180.0, "min_us": 650, "max_us": 2350,
     "max_speed": 120.0, "max_accel": 260.0},  # S1
    {"pin": 16, "mode": "positional", "range_deg": 180.0, "min_us": 650, "max_us": 2350,
     "max_speed": 120.0, "max_accel": 260.0},  # S2
    {"pin": 26, "mode": "positional", "range_deg": 180.0, "min_us": 650, "max_us": 2350,
     "max_speed": 120.0, "max_accel": 260.0},  # S3

    # ---- Continuous rotation servos (speed control) ----
    # neutral_us: ค่าหยุด (ส่วนมาก ~1500us)
    # max_delta_us: จำกัดความแรงสูงสุด (ยิ่งน้อยยิ่งหมุนเบา)
    # deadband_us: ช่วงกันสั่นใกล้ neutral
    # slew_us_per_s: อัตราไต่ของ PWM (ยิ่งน้อยยิ่งค่อยๆออกตัว ลดกระชากตอนเริ่มหมุน)
    {"pin": 20, "mode": "positional", "range_deg": 180.0, "min_us": 650, "max_us": 2350, "max_speed": 80.0, "max_accel": 180.0},  # S4 (0-180°)
    {"pin": 21, "mode": "positional", "range_deg": 180.0, "min_us": 650, "max_us": 2350, "max_speed": 80.0, "max_accel": 180.0},  # S5 (0-180°)
]

SERVO_UPDATE_SEC = 0.02      # 50Hz (อัปเดตคำสั่ง)
SNAP_EPS_DEG = 0.20          # positional: ใกล้เป้าหมายเท่านี้ให้ snap เข้าเป้า

# UI step
ANGLE_STEP_INIT = 2.0        # สำหรับ positional
MIN_ANGLE_STEP = 0.5
MAX_ANGLE_STEP = 20.0

SPEED_STEP_INIT = 0.05       # สำหรับ continuous (-1..1)
MIN_SPEED_STEP = 0.01
MAX_SPEED_STEP = 0.30

SAVE_FILE = "pose_5servos.json"

DETACH_ON_IDLE = True
IDLE_SECONDS = 0.35

HOLD_BETWEEN_KEYS = 0.15
# ==========================


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def angle_to_us(cfg, angle):
    rng = float(cfg["range_deg"])
    a = clamp(float(angle), 0.0, rng)
    min_us = int(cfg["min_us"])
    max_us = int(cfg["max_us"])
    return int(min_us + (a / rng) * (max_us - min_us))


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

    pins = [c["pin"] for c in SERVO_CONFIG]
    n = len(pins)

    for pin in pins:
        try:
            lgpio.gpio_claim_output(h, pin)
        except Exception:
            pass

    def set_servo_us(pin, us):
        lgpio.tx_servo(h, pin, int(us))

    poses = load_poses()

    selected = 0  # 0..4 แสดง servo 1..5
    angle_step = ANGLE_STEP_INIT
    speed_step = SPEED_STEP_INIT

    # ---- state (positional) ----
    current_deg = [0.0] * n
    target_deg = [0.0] * n
    vel_deg_s = [0.0] * n

    # ---- state (continuous) ----
    # speed: -1..1
    current_speed = [0.0] * n
    target_speed = [0.0] * n
    current_us = [0] * n
    target_us = [0] * n

    pulses_active = [True] * n

    def cfg(i):
        return SERVO_CONFIG[i]

    def is_cont(i):
        return cfg(i).get("mode") == "continuous"

    def is_pos(i):
        return not is_cont(i)

    def neutral_us(i):
        return int(cfg(i).get("neutral_us", 1500))

    def speed_to_pulse(i, speed01):
        c = cfg(i)
        neu = int(c.get("neutral_us", 1500))
        max_delta = int(c.get("max_delta_us", 200))

        s = clamp(float(speed01), -1.0, 1.0)

        # expo ทำให้ออกตัวนิ่ม: ใกล้ 0 จะเบามากขึ้น (ลดกระชากตอนเริ่มหมุน)
        s = (abs(s) ** 3) * (1.0 if s >= 0 else -1.0)

        return neu + int(s * max_delta)

    def set_target_speed(i, speed01):
        target_speed[i] = clamp(float(speed01), -1.0, 1.0)
        target_us[i] = speed_to_pulse(i, target_speed[i])

    # init: set to 0° for positional and neutral for continuous
    for i, pin in enumerate(pins):
        if is_cont(i):
            u = neutral_us(i)
            current_us[i] = u
            target_us[i] = u
            set_servo_us(pin, u)
        else:
            u = angle_to_us(cfg(i), 0.0)
            set_servo_us(pin, u)

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

        saved = len(poses)
        max_cols = max(1, curses.COLS - 1)
        max_rows = curses.LINES

        # header
        sel_mode = "CONT" if is_cont(selected) else "POS"
        step_txt = f"{speed_step:.2f}" if is_cont(selected) else f"{angle_step:.1f}°"
        lines = [
            f"Sel: {selected+1} ({sel_mode})   saved: {saved}   step: {step_txt}",
        ]

        # servo lines
        for i in range(n):
            tag = f"S{i+1}"
            if is_cont(i):
                neu = neutral_us(i)
                lines.append(
                    f"{tag:<2}  speed {current_speed[i]:+5.2f} -> {target_speed[i]:+5.2f}   "
                    f"us {current_us[i]} -> {target_us[i]}   (neu {neu})"
                )
            else:
                rng = cfg(i)["range_deg"]
                lines.append(f"{tag:<2}  {current_deg[i]:7.1f} -> {target_deg[i]:7.1f}   (0-{rng:g}°)")

        # help
        lines.append("1-5 select | POS: <-/+  ->/- | CONT: <- faster  -> slower | UP/DN step | x=stop(cont) s=save l=list p=play h=home c=clear q=quit")

        if status_msg:
            lines.append(status_msg)

        # render
        row = 0
        for line in lines:
            if row >= max_rows:
                break
            stdscr.addstr(row, 0, line[:max_cols])
            row += 1

        # log area
        start_row = row + 1
        max_log_rows = max_rows - start_row
        if max_log_rows > 0:
            tail = log_lines[-max_log_rows:]
            for i, line in enumerate(tail):
                stdscr.addstr(start_row + i, 0, line[:max_cols])

        stdscr.refresh()

    def ensure_pulses_on(i):
        if not pulses_active[i]:
            if is_cont(i):
                # re-arm with current pulse (or neutral if unknown)
                u = current_us[i] if current_us[i] else neutral_us(i)
                set_servo_us(pins[i], u)
            else:
                set_servo_us(pins[i], angle_to_us(cfg(i), current_deg[i]))
            pulses_active[i] = True

    def set_target_angle(i, angle):
        rng = float(cfg(i)["range_deg"])
        target_deg[i] = clamp(float(angle), 0.0, rng)

    def step_positional_smooth(i, dt):
        """positional: ขยับแบบจำกัดความเร็ว+ความเร่ง (accel-limited) ในหน่วยองศา"""
        err = target_deg[i] - current_deg[i]

        if abs(err) <= SNAP_EPS_DEG:
            current_deg[i] = target_deg[i]
            vel_deg_s[i] = 0.0
            set_servo_us(pins[i], angle_to_us(cfg(i), current_deg[i]))
            return

        c = cfg(i)
        max_speed = float(c.get("max_speed", 120.0))
        max_accel = float(c.get("max_accel", 260.0))

        desired_v = clamp(err / max(dt, 1e-6), -max_speed, max_speed)

        dv = desired_v - vel_deg_s[i]
        max_dv = max_accel * dt  # <-- สำคัญ: ใช้ max_accel ของตัวนั้น
        dv = clamp(dv, -max_dv, max_dv)
        vel_deg_s[i] += dv

        nxt = current_deg[i] + vel_deg_s[i] * dt

        # กัน overshoot
        if (err > 0 and nxt > target_deg[i]) or (err < 0 and nxt < target_deg[i]):
            nxt = target_deg[i]
            vel_deg_s[i] = 0.0

        current_deg[i] = nxt
        set_servo_us(pins[i], angle_to_us(cfg(i), current_deg[i]))

    def step_continuous_smooth(i, dt):
        """continuous: ไต่ค่า PWM (us) แบบ ramp + deadband เพื่อกันออกตัวแรง/กระชาก"""
        c = cfg(i)
        neu = int(c.get("neutral_us", 1500))
        dead = int(c.get("deadband_us", 12))
        slew = float(c.get("slew_us_per_s", 140.0))

        # จำกัดการเปลี่ยนค่า PWM ต่อรอบ (slew-rate)
        max_step = int(max(1, slew * dt))
        diff = target_us[i] - current_us[i]
        if abs(diff) <= max_step:
            current_us[i] = target_us[i]
        else:
            current_us[i] += max_step if diff > 0 else -max_step

        # deadband รอบ neutral: กันสั่นและกันเริ่มหมุนจาก noise
        if abs(current_us[i] - neu) <= dead:
            current_us[i] = neu

        set_servo_us(pins[i], current_us[i])

        # อัปเดต current_speed ให้แสดงผล (ประมาณจาก target_speed)
        # (ไม่ใช่ feedback จริง แต่พอใช้แสดงใน UI)
        current_speed[i] = target_speed[i] if current_us[i] == target_us[i] else current_speed[i]

    def all_reached():
        # positional: ถึงเป้ามุม / continuous: ถึงเป้า PWM
        for i in range(n):
            if is_cont(i):
                if abs(current_us[i] - target_us[i]) > 1:
                    return False
            else:
                if abs(current_deg[i] - target_deg[i]) > SNAP_EPS_DEG:
                    return False
        return True

    def idle_stop_if_needed():
        if not DETACH_ON_IDLE:
            return
        if not all_reached():
            return
        if (time.time() - last_input_time) <= IDLE_SECONDS:
            return

        for i, pin in enumerate(pins):
            if pulses_active[i]:
                ok = stop_pulses_safe(h, pin)
                pulses_active[i] = False if ok else True

    def wait_until_reached(timeout_sec=8.0):
        nonlocal last_servo_update
        start = time.time()
        while not all_reached() and (time.time() - start) < timeout_sec:
            now = time.time()
            if (now - last_servo_update) >= SERVO_UPDATE_SEC:
                dt = now - last_servo_update
                for i in range(n):
                    if pulses_active[i]:
                        if is_cont(i):
                            step_continuous_smooth(i, dt)
                        else:
                            step_positional_smooth(i, dt)
                last_servo_update = now
            time.sleep(0.002)

    def save_pose():
        # เก็บแบบ "command" รายตัว: positional เก็บองศา, continuous เก็บ speed (-1..1)
        cmd = []
        for i in range(n):
            if is_cont(i):
                cmd.append(round(target_speed[i], 3))
            else:
                cmd.append(round(target_deg[i], 2))

        poses.append({"t": round(time.time(), 3), "cmd": cmd})
        save_poses(poses)
        userlog(f"saved pose {len(poses):02d}: {cmd}")

    def list_poses():
        if not poses:
            userlog("(no poses)")
            return
        userlog("=== Poses ===")
        for i, e in enumerate(poses, 1):
            if "cmd" in e:
                userlog(f"{i:02d}) {e.get('cmd')}")
            else:
                # รองรับไฟล์เก่า (angles)
                userlog(f"{i:02d}) {e.get('angles')}")
        userlog("=== end ===")

    def playback():
        if not poses:
            userlog("No poses to play.")
            return
        userlog(f"=== Playback ({len(poses)} poses) ===")

        for k, e in enumerate(poses, 1):
            # รองรับไฟล์เก่า: angles
            if "cmd" in e:
                cmd = e.get("cmd", [0] * n)
            else:
                cmd = e.get("angles", [0] * n)

            for i in range(n):
                ensure_pulses_on(i)
                if is_cont(i):
                    # ถ้าไฟล์เก่าเป็น "องศา" มา จะ clamp ให้อยู่ใน -1..1 (มักจะใหญ่) -> กลายเป็นแรงมาก
                    # เลย treat ว่า continuous ในไฟล์เก่า = 0 (หยุด) เพื่อความปลอดภัย
                    if "cmd" in e:
                        set_target_speed(i, clamp(float(cmd[i]), -1.0, 1.0))
                    else:
                        set_target_speed(i, 0.0)
                else:
                    rng = float(cfg(i)["range_deg"])
                    set_target_angle(i, clamp(float(cmd[i]), 0.0, rng))

            userlog(f"[{k:02d}/{len(poses):02d}] -> {cmd}")
            draw("PLAYING...")
            wait_until_reached(timeout_sec=10.0)
            time.sleep(HOLD_BETWEEN_KEYS)

        userlog("=== Playback done ===")

    # initial draw
    draw()

    try:
        while True:
            key = stdscr.getch()
            if key != -1:
                last_input_time = time.time()

                # เลือกเซอร์โวด้วยเลข 1-5
                if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                    selected = key - ord('1')
                    selected = clamp(selected, 0, n - 1)

                elif key == ord('q'):
                    break

                # คุม:
                # - POS: ซ้ายเพิ่มองศา / ขวาลดองศา
                # - CONT: ซ้ายเพิ่มความเร็ว / ขวาลดความเร็ว
                elif key == curses.KEY_LEFT:
                    ensure_pulses_on(selected)
                    if is_cont(selected):
                        target_speed[selected] = clamp(target_speed[selected] + speed_step, -1.0, 1.0)
                        set_target_speed(selected, target_speed[selected])
                    else:
                        set_target_angle(selected, target_deg[selected] + angle_step)

                elif key == curses.KEY_RIGHT:
                    ensure_pulses_on(selected)
                    if is_cont(selected):
                        target_speed[selected] = clamp(target_speed[selected] - speed_step, -1.0, 1.0)
                        set_target_speed(selected, target_speed[selected])
                    else:
                        set_target_angle(selected, target_deg[selected] - angle_step)

                elif key == curses.KEY_UP:
                    # เพิ่ม step ตามโหมดที่เลือกอยู่
                    if is_cont(selected):
                        speed_step = clamp(speed_step + 0.01, MIN_SPEED_STEP, MAX_SPEED_STEP)
                    else:
                        angle_step = clamp(angle_step + 0.5, MIN_ANGLE_STEP, MAX_ANGLE_STEP)

                elif key == curses.KEY_DOWN:
                    if is_cont(selected):
                        speed_step = clamp(speed_step - 0.01, MIN_SPEED_STEP, MAX_SPEED_STEP)
                    else:
                        angle_step = clamp(angle_step - 0.5, MIN_ANGLE_STEP, MAX_ANGLE_STEP)

                elif key == ord('x'):
                    # หยุดสำหรับ continuous
                    if is_cont(selected):
                        ensure_pulses_on(selected)
                        set_target_speed(selected, 0.0)

                elif key == ord('h'):
                    ensure_pulses_on(selected)
                    if is_cont(selected):
                        set_target_speed(selected, 0.0)
                    else:
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
                dt = now - last_servo_update
                for i in range(n):
                    if pulses_active[i]:
                        if is_cont(i):
                            step_continuous_smooth(i, dt)
                        else:
                            step_positional_smooth(i, dt)
                last_servo_update = now

            idle_stop_if_needed()
            draw()
            time.sleep(0.002)

    finally:
        for i, pin in enumerate(pins):
            try:
                # ปิดให้เป็น neutral/0 ก่อนหยุดพัลส์ เพื่อลดอาการกระชากตอนออก
                if is_cont(i):
                    set_servo_us(pin, neutral_us(i))
                    time.sleep(0.03)
                stop_pulses_safe(h, pin)
            except Exception:
                pass
        lgpio.gpiochip_close(h)


if __name__ == "__main__":
    curses.wrapper(run)
