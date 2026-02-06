import time
import random
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import serial

# ===================== CONFIG =====================
FRAME_W, FRAME_H = 1280, 720
SHOW = True

# --- Models (AI2: 2 models) ---
PERSON_MODEL_PATH = "yolov8n.pt"   # detect person
BOOK_MODEL_PATH   = "best.pt"      # detect book

# person in COCO is class 0
PERSON_ID = 0

# book model class: auto-detect by name; fallback to 0 (common for custom 1-class model)
BOOK_CLASS_NAME = "book"
BOOK_ID_FALLBACK = 0

CONF_PERSON = 0.75
CONF_BOOK   = 0.50

MIN_AREA_RATIO_PERSON = 0.03
MIN_AREA_RATIO_BOOK   = 0.02

# Camera color: prefer BGR (OpenCV native). If not supported, fall back to RGB + convert.
CAMERA_PREFER_FORMAT = "BGR888"   # try first
CAMERA_FALLBACK_FORMAT = "RGB888"  # if above fails

INIT_IDLE_SEC = 3
CONFIRM_PERSON_SEC = 2
REST_SEC = 5
BOOK_MODE_SEC = 10

SCAN_MIN = 15
SCAN_MAX = 165
SCAN_SPEED = 35

POSE_TMS = 1500
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
# ===================== /CONFIG =====================


def _read_lines(ser, max_ms=200):
    """อ่านบรรทัดจาก UNO ภายในเวลาที่กำหนด"""
    end = time.time() + (max_ms / 1000.0)
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
    """ส่งคำสั่งแล้วรอ OK/ERR (return True/False)"""
    cmd = cmd.strip()
    for _ in range(retry + 1):
        if DEBUG_SERIAL:
            print(">>", cmd)
        ser.write((cmd + "\n").encode())
        ser.flush()
        time.sleep(0.015)

        lines = _read_lines(ser, max_ms=wait_ms)
        if DEBUG_SERIAL and lines:
            for s in lines:
                print("<<", s)

        if any(s == "OK" for s in lines):
            return True
        if any(s == "ERR" for s in lines):
            time.sleep(0.05)
            continue

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


def _area_ratio(x1, y1, x2, y2, frame_area):
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (area / frame_area) if frame_area > 0 else 0.0


def _boxes_from_res(res, label_map=None):
    """Return list of (x1,y1,x2,y2,conf,label,cls_id)"""
    out = []
    if res.boxes is None:
        return out
    for b in res.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if label_map and cls_id in label_map:
            label = label_map[cls_id]
        else:
            label = str(cls_id)
        out.append((x1, y1, x2, y2, conf, label, cls_id))
    return out


def pick_book_class_id(book_model):
    """หา class id ของหนังสือจาก model.names แบบอัตโนมัติ"""
    names = getattr(book_model, "names", None)
    if isinstance(names, dict):
        # exact match
        for k, v in names.items():
            if str(v).lower() == BOOK_CLASS_NAME.lower():
                return int(k)
        # if single class model, just use that
        if len(names) == 1:
            return int(next(iter(names.keys())))
    return BOOK_ID_FALLBACK


def main():
    # ---------- UNO ----------
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1, write_timeout=0.2)
    time.sleep(2.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    _read_lines(ser, max_ms=200)

    ok = set_light(ser, "WHITE")
    if not ok:
        print("[WARN] UNO ไม่ตอบ OK (อาจยัง OFF / ยังไม่กดปุ่ม ON / หรือพอร์ตไม่ถูก)")
    hold_all(ser)

    # ---------- Camera ----------
    cam = Picamera2()
    cam_format = CAMERA_PREFER_FORMAT
    need_rgb2bgr = False

    try:
        cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": cam_format})
        cam.configure(cfg)
    except Exception as e:
        # fallback
        print(f"[WARN] Camera format {CAMERA_PREFER_FORMAT} ใช้ไม่ได้ ({e}). fallback -> {CAMERA_FALLBACK_FORMAT}")
        cam_format = CAMERA_FALLBACK_FORMAT
        cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": cam_format})
        cam.configure(cfg)
        need_rgb2bgr = True

    cam.start()
    time.sleep(0.2)

    # ---------- YOLO (2 models) ----------
    person_model = YOLO(PERSON_MODEL_PATH)
    book_model = YOLO(BOOK_MODEL_PATH)

    book_id = pick_book_class_id(book_model)

    # show class maps once
    try:
        print("[PERSON names]", person_model.names)
    except Exception:
        pass
    try:
        print("[BOOK names]", book_model.names)
    except Exception:
        pass
    print(f"[TARGET] PERSON_ID={PERSON_ID} (YOLOv8n), BOOK_ID={book_id} (best.pt)")

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
            frame = cam.capture_array()

            # Color fix
            if need_rgb2bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w = frame.shape[:2]
            frame_area = w * h

            # --- detections ---
            person_res = person_model(frame, classes=[PERSON_ID], conf=0.25, verbose=False)[0]
            book_res = book_model(frame, classes=[book_id], conf=0.25, verbose=False)[0]

            person_boxes = _boxes_from_res(person_res, label_map={PERSON_ID: "person"})
            book_boxes = _boxes_from_res(book_res, label_map={book_id: "book"})

            person_found = False
            for (x1, y1, x2, y2, conf, _label, cls_id) in person_boxes:
                if cls_id == PERSON_ID and conf >= CONF_PERSON and _area_ratio(x1, y1, x2, y2, frame_area) >= MIN_AREA_RATIO_PERSON:
                    person_found = True
                    break

            book_found = False
            for (x1, y1, x2, y2, conf, _label, cls_id) in book_boxes:
                if cls_id == book_id and conf >= CONF_BOOK and _area_ratio(x1, y1, x2, y2, frame_area) >= MIN_AREA_RATIO_BOOK:
                    book_found = True
                    break

            boxes = [(x1, y1, x2, y2, conf, label) for (x1, y1, x2, y2, conf, label, _cls) in (person_boxes + book_boxes)]

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

            # -------- draw --------
            for (x1, y1, x2, y2, conf, label) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"STATE: {state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"person={person_found} book={book_found}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
        except Exception:
            pass


if __name__ == "__main__":
    main()
