import lgpio
import time
import json
import os

# ===== CONFIG =====
SERVO_GPIO = 12          # ใช้ GPIO12 (Pin 32)
MIN_US = 600             # MG90S มักนิ่งกว่าช่วง 600-2400
MAX_US = 2400
CENTER_US = 1500

STEP_US = 5              # ยิ่งน้อยยิ่งเนียน แต่ช้าลง
STEP_DELAY = 0.01        # หน่วงต่อ step

HOLD_TIME = 0.90         # ถึงมุมแล้ว "ถือ" สักพักให้เข้าที่
DETACH_AFTER_MOVE = True # True = ถึงแล้วปล่อย (นิ่ง/ไม่สั่น) | False = ล็อกมุม

SAVE_FILE = "pose.json"
# ==================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def angle_to_us(angle):
    angle = clamp(float(angle), 0.0, 180.0)
    return int(MIN_US + (angle / 180.0) * (MAX_US - MIN_US))

def us_to_angle(us):
    us = clamp(float(us), MIN_US, MAX_US)
    return (us - MIN_US) * 180.0 / (MAX_US - MIN_US)

# ---- GPIO setup ----
h = lgpio.gpiochip_open(0)
try:
    lgpio.gpio_claim_output(h, SERVO_GPIO)
except Exception:
    pass

current_us = CENTER_US

def set_servo_us(pulse_us):
    # pulse_us = 0 -> stop pulses (detach)
    lgpio.tx_servo(h, SERVO_GPIO, int(pulse_us))

def detach():
    set_servo_us(0)

def move_to_angle(angle, smooth=True, hold_time=HOLD_TIME, detach_after=DETACH_AFTER_MOVE):
    global current_us
    target_us = angle_to_us(angle)

    if not smooth:
        current_us = target_us
        set_servo_us(current_us)
        if hold_time > 0:
            time.sleep(hold_time)
        if detach_after:
            detach()
        return

    while abs(current_us - target_us) > STEP_US:
        current_us += STEP_US if current_us < target_us else -STEP_US
        set_servo_us(current_us)
        time.sleep(STEP_DELAY)

    current_us = target_us
    set_servo_us(current_us)

    if hold_time > 0:
        time.sleep(hold_time)
    if detach_after:
        detach()

# ---- load/save poses ----
poses = []
if os.path.exists(SAVE_FILE):
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            poses = json.load(f)
        if not isinstance(poses, list):
            poses = []
    except Exception:
        poses = []

t0 = time.time()

def save_poses():
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(poses, f, ensure_ascii=False, indent=2)

def print_help():
    print("\nคำสั่ง:")
    print("  0-180  -> สั่งไปมุมที่พิมพ์ (องศา)")
    print("  s      -> บันทึกมุมปัจจุบันเป็น keyframe")
    print("  p      -> เล่นกลับทั้งหมด (playback)")
    print("  h      -> กลับ 0° (home)")
    print("  l      -> แสดงรายการที่บันทึกไว้")
    print("  c      -> ล้างรายการทั้งหมด")
    print("  q      -> ออก")
    print("  cfg    -> แสดง config ปัจจุบัน\n")

def print_cfg():
    print("\nCONFIG:")
    print(f"  GPIO={SERVO_GPIO}")
    print(f"  MIN_US={MIN_US}  MAX_US={MAX_US}  CENTER_US={CENTER_US}")
    print(f"  STEP_US={STEP_US}  STEP_DELAY={STEP_DELAY}")
    print(f"  HOLD_TIME={HOLD_TIME}")
    print(f"  DETACH_AFTER_MOVE={DETACH_AFTER_MOVE}")
    print(f"  SAVE_FILE={SAVE_FILE}\n")

print("=== Teach & Playback (lgpio.tx_servo) ===")
print_cfg()
print_help()

# เริ่มที่ 0° (เปลี่ยนเป็น 90° ก็ได้)
move_to_angle(0, smooth=False)

try:
    while True:
        cmd = input(">> ").strip().lower()

        if cmd in ("?", "help"):
            print_help()
            continue

        if cmd == "cfg":
            print_cfg()
            continue

        if cmd == "q":
            break

        if cmd == "h":
            print("Home: 0°")
            move_to_angle(0, smooth=True)
            continue

        if cmd == "s":
            angle_now = us_to_angle(current_us)
            entry = {
                "t": round(time.time() - t0, 3),  # ไว้ต่อยอดแบบตามเวลา
                "angle": round(angle_now, 2)
            }
            poses.append(entry)
            save_poses()
            print(f"บันทึก: {entry['angle']}° (รวม {len(poses)} จุด)")
            continue

        if cmd == "l":
            if not poses:
                print("ยังไม่มี keyframe")
            else:
                for i, e in enumerate(poses, 1):
                    print(f"{i:02d}) {e['angle']}°  (t={e['t']}s)")
            continue

        if cmd == "c":
            poses.clear()
            save_poses()
            print("ล้างรายการแล้ว")
            continue

        if cmd == "p":
            if not poses:
                print("ยังไม่มี keyframe (พิมพ์องศาแล้วกด s เพื่อบันทึก)")
                continue

            print(f"Play: {len(poses)} จุด")
            for e in poses:
                a = float(e["angle"])
                print(f"-> {a:.1f}°")
                move_to_angle(a, smooth=True)
                time.sleep(0.2)  # หน่วงระหว่างท่า ปรับได้
            print("จบการเล่นกลับ")
            continue

        # ตีความเป็นองศา
        try:
            a = float(cmd)
            a = clamp(a, 0.0, 180.0)
            print(f"ไปที่ {a:.1f}°")
            move_to_angle(a, smooth=True)
        except ValueError:
            print("คำสั่งไม่ถูกต้อง (พิมพ์ ? เพื่อดู help)")

finally:
    detach()
    lgpio.gpiochip_close(h)
    print("ออกจากโปรแกรมแล้ว")
