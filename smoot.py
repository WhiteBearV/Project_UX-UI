import time
import random
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import serial

FRAME_W, FRAME_H = 1280, 720
SHOW = True

PERSON_ID = 0
BOOK_ID = 73

MODEL_PATH = "yolo11n.pt"
CONF_PERSON = 0.55
CONF_BOOK = 0.30
MIN_AREA_RATIO_PERSON = 0.03
MIN_AREA_RATIO_BOOK   = 0.02

INIT_IDLE_SEC = 3
CONFIRM_PERSON_SEC = 2
REST_SEC = 5
BOOK_MODE_SEC = 10

SCAN_MIN = 15
SCAN_MAX = 165
SCAN_SPEED = 35

POSE_TMS = 1500
# เผื่อเวลาให้เซอร์โว "ถึงเป้า" จริงก่อนเข้า REST (กัน HOLD ตัดกลางทางแล้วสะบัด)
POSE_EXTRA_WAIT = 0.8

POSES = [
    (90, 130, 0, 110, 180),
    (0, 0, 0, 90,  0),
    (74, 0, 52, 0,  180),
    (0, 0, 0, 72, 0),
    (0, 0, 0, 0,  180),
]

LAMP_NORMAL = (180, 60, 90, 180)

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

DEBUG_SERIAL = True

def _read_lines(ser, max_ms=200):
    """อ่านบรรทัดจาก UNO ภายในเวลาที่กำหนด"""
    end = time.time() + (max_ms/1000.0)
    lines = []
    while time.time() < end:
        line = ser.readline()
        if not line:
            continue
        s = line.decode(errors="ignore").strip()
        if s:
            lines.append(s)
    return lines

def send_wait(ser, cmd: str, wait_ms=250, retry=1):
    """
    ส่งคำสั่งแล้วรอ OK/ERR
    return: True/False
    """
    cmd = cmd.strip()
    for _ in range(retry + 1):
        if DEBUG_SERIAL:
            print(">>", cmd)
        ser.write((cmd + "\n").encode())
        ser.flush()
        time.sleep(0.015)  # กันส่งรัวเกิน

        lines = _read_lines(ser, max_ms=wait_ms)
        if DEBUG_SERIAL and lines:
            for s in lines:
                print("<<", s)

        # ถ้าเจอ OK ถือว่าสำเร็จ
        if any(s == "OK" for s in lines):
            return True
        # ถ้าเจอ ERR ให้ลองส่งซ้ำ (บางที UNO ยังไม่ armed)
        if any(s == "ERR" for s in lines):
            time.sleep(0.05)
            continue

        # ไม่ตอบเลย
        time.sleep(0.05)

    return False

def set_light(ser, mode: str):
    return send_wait(ser, f"LIGHT {mode}")

def hold_all(ser):
    return send_wait(ser, "HOLD ALL")

def start_scan(ser):
    set_light(ser, "RED")
    return send_wait(ser, f"SCAN S1 {SCAN_MIN} {SCAN_MAX} {SCAN_SPEED}")

def stop_scan(ser):
    return send_wait(ser, "STOPSCAN")

def book_mode_on(ser):
    stop_scan(ser)
    send_wait(ser, f"LAMP_NORMAL {LAMP_NORMAL[0]} {LAMP_NORMAL[1]} {LAMP_NORMAL[2]} {LAMP_NORMAL[3]}")
    send_wait(ser, "BOOK_HOLD ON")
    set_light(ser, "WHITE")

def book_mode_off(ser):
    return send_wait(ser, "BOOK_HOLD OFF")

_last_pose_idx = -1
def choose_pose():
    global _last_pose_idx
    idx = random.randrange(len(POSES))
    if len(POSES) > 1:
        while idx == _last_pose_idx:
            idx = random.randrange(len(POSES))
    _last_pose_idx = idx
    return POSES[idx]

def detect_flags_from_result(res, frame_area):
    person_found = False
    book_found = False
    boxes = []

    if res.boxes is None:
        return person_found, book_found, boxes

    for b in res.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_ratio = area / frame_area if frame_area > 0 else 0.0

        label = "person" if cls_id == PERSON_ID else ("book" if cls_id == BOOK_ID else str(cls_id))
        boxes.append((x1, y1, x2, y2, conf, label))

        if cls_id == PERSON_ID and conf >= CONF_PERSON and area_ratio >= MIN_AREA_RATIO_PERSON:
            person_found = True
        if cls_id == BOOK_ID and conf >= CONF_BOOK and area_ratio >= MIN_AREA_RATIO_BOOK:
            book_found = True

    return person_found, book_found, boxes

def main():
    # ---------- UNO ----------
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1, write_timeout=0.2)
    time.sleep(2.2)  # ให้ UNO รีเซ็ตให้เสร็จก่อน
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # อ่าน READY/ข้อความค้าง (ถ้ามี)
    _read_lines(ser, max_ms=200)

    # *** สำคัญ ***
    # ถ้า UNO เป็นแบบปุ่ม toggle: ต้องกด ON ที่ปุ่มก่อน
    # ที่นี่เราจะลองส่ง LIGHT WHITE ถ้าไม่ได้ OK จะพิมพ์เตือน
    ok = set_light(ser, "WHITE")
    if not ok:
        print("[WARN] UNO ไม่ตอบ OK (อาจยัง OFF / ยังไม่กดปุ่ม ON / หรือพอร์ตไม่ถูก)")
    hold_all(ser)

    # ---------- Camera ----------
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

    # ---------- YOLO ----------
    model = YOLO(MODEL_PATH)

    # ---------- State ----------
    state = "INIT"
    state_until = time.time() + INIT_IDLE_SEC
    book_until = None
    scan_started = False
    confirm_start = None

    t0 = time.time()
    fps = 0.0

    try:
        while True:
            rgb = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            frame_area = w * h

            res = model(frame, classes=[PERSON_ID, BOOK_ID], conf=0.25, verbose=False)[0]
            person_found, book_found, boxes = detect_flags_from_result(res, frame_area)

            now = time.time()

            if state == "INIT":
                if now >= state_until:
                    state = "SCAN"
                    scan_started = False
                    confirm_start = None

            elif state == "SCAN":
                if book_found:
                    book_mode_on(ser)
                    state = "BOOK"
                    book_until = now + BOOK_MODE_SEC
                    scan_started = False
                    confirm_start = None
                else:
                    if not scan_started:
                        start_scan(ser)
                        scan_started = True

                    if person_found:
                        stop_scan(ser)
                        state = "CONFIRM"
                        confirm_start = now

            elif state == "CONFIRM":
                if book_found:
                    book_mode_on(ser)
                    state = "BOOK"
                    book_until = now + BOOK_MODE_SEC
                    confirm_start = None
                else:
                    if not person_found:
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None
                    else:
                        if confirm_start is not None and (now - confirm_start) >= CONFIRM_PERSON_SEC:
                            set_light(ser, "SUNRISE")
                            pose = choose_pose()
                            send_wait(ser, f"POSE {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {POSE_TMS}")
                            state = "POSE"
                            state_until = now + (POSE_TMS / 1000.0) + POSE_EXTRA_WAIT
                            confirm_start = None

            elif state == "POSE":
                if now >= state_until:
                    hold_all(ser)
                    set_light(ser, "BREATH")
                    state = "REST"
                    state_until = now + REST_SEC

            elif state == "REST":
                if now >= state_until:
                    state = "SCAN"
                    scan_started = False

            elif state == "BOOK":
                if now >= book_until:
                    book_mode_off(ser)
                    hold_all(ser)
                    state = "SCAN"
                    scan_started = False

            # -------- draw ----------
            for (x1, y1, x2, y2, conf, label) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"STATE: {state}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"person={person_found} book={book_found}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, FRAME_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if SHOW:
                cv2.imshow("PiCam YOLO + UNO Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    stop_scan(ser)
                    book_mode_off(ser)
                    send_wait(ser, "HOME", wait_ms=400, retry=2)
                    time.sleep(POSE_TMS / 1000.0 + 0.3)
                    hold_all(ser)
                    set_light(ser, "OFF")
                    break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        try:
            hold_all(ser)
            set_light(ser, "WHITE")
        except:
            pass

if __name__ == "__main__":
    main()
