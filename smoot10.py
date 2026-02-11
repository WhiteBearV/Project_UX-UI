import time
import random
import cv2
import serial
import os
import threading
from picamera2 import Picamera2
from ultralytics import YOLO


class Config:
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 115200
    SERIAL_TIMEOUT = 0.03  # ลดหน่วงอ่าน serial

    FRAME_W = 640
    FRAME_H = 360
    CAMERA_FORMAT = "BGR888"
    SHOW_WINDOW = True  # ถ้า SSH ไม่มีจอ จะปิดเองอัตโนมัติ

    # --- Performance ---
    # UI FPS จะมาจากกล้องเป็นหลัก ส่วน AI จะรันตามช่วงเวลา
    INFER_IMGSZ = 416          # ลดจาก 640 เพื่อให้ CPU เร็วขึ้นมาก
    AI_PERIOD_SEC = 0.20       # รัน AI ทุก 0.20s (~5Hz) แต่ UI ยังวิ่ง >12 FPS ได้
    STATUS_PERIOD_SEC = 0.50   # ขอ STATUS ทุก 0.5s พอ (อย่าถามทุกเฟรม)

    CONF_PERSON = 0.60
    CONF_BOOK = 0.20
    MIN_AREA_PERSON = 0.03
    MIN_AREA_BOOK = 0.01

    MODEL_PATH = "yolov8n.pt"  # ใช้โมเดลเดียว ตรวจทั้งคน+หนังสือ
    PERSON_CLASS_ID = 0
    BOOK_CLASS_NAME = "book"

    CONFIRM_PERSON_SEC = 2.0
    NO_DETECTION_TIMEOUT = 10.0
    DEEP_SLEEP_DURATION = 10.0

    BOOK_MODE_SEC = 15.0
    HOLD_POSE_SEC = 2.0
    REST_SEC = 3.0

    INIT_POSE = (0, 0, 0, 0, 0)
    REST_POSE = (0, 0, 0, 0, 90)

    SCAN_MIN = 15
    SCAN_MAX = 165
    SCAN_SPEED = 20

    POSE_TIME_MS = 1200

    POSES = [
        (90, 130, 0, 110, 180),
        (0, 0, 0, 90,  0),
        (74, 0, 52, 0,  180),
        (0, 0, 180, 72, 0),
        (90, 0, 0, 90,  180),
        (120, 90, 0, 90, 90),
        (60, 180, 60, 130, 180),
        (150, 90, 0, 90, 0),
    ]

    BOOK_LIGHT = "WHITE"
    BOOK_POSE = (90, 180, 60, 90, 90)

    POSE_SETTLE_SEC = 0.15
    POLL_EVENT_EVERY = 0.05


def wait_pose_done():
    time.sleep((Config.POSE_TIME_MS / 1000.0) + Config.POSE_SETTLE_SEC)


class RobotArm:
    def __init__(self):
        self.ser = serial.Serial(
            Config.SERIAL_PORT, Config.BAUD_RATE,
            timeout=Config.SERIAL_TIMEOUT, write_timeout=0.1
        )
        time.sleep(1.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self._last_poll = 0.0
        self._seen_shutdown_begin = False
        self._seen_shutdown_done = False

        # cache armed status
        self._armed_cache = False
        self._last_status = 0.0

    def _read_lines(self, max_ms=120):
        end = time.time() + (max_ms / 1000.0)
        out = []
        while time.time() < end:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        out.append(line)
                except:
                    pass
            else:
                time.sleep(0.001)
        return out

    def cmd(self, s, wait_ok=True):
        s = s.strip()
        try:
            self.ser.write((s + "\n").encode())
            self.ser.flush()
        except:
            return False, ["ERR_SERIAL_WRITE"]

        if not wait_ok:
            return True, []

        lines = self._read_lines(160)

        for ln in lines:
            if "EVENT SHUTDOWN_BEGIN" in ln:
                self._seen_shutdown_begin = True
            if "EVENT SHUTDOWN_DONE" in ln:
                self._seen_shutdown_done = True

        if any("ERR_NOT_ARMED" in x for x in lines):
            return False, lines
        if any("OK" in x for x in lines):
            return True, lines
        if any("ERR" in x for x in lines):
            return False, lines
        return False, lines

    def status(self):
        ok, lines = self.cmd("STATUS", wait_ok=True)
        for ln in lines:
            if ln.startswith("STATUS"):
                return ln
        return ""

    def is_armed_fast(self):
        # ไม่ถาม STATUS ทุกเฟรม ลด latency
        now = time.time()
        if (now - self._last_status) >= Config.STATUS_PERIOD_SEC:
            self._last_status = now
            st = self.status()
            if st:
                self._armed_cache = ("ARMED=1" in st)
        return self._armed_cache

    def light(self, mode): return self.cmd(f"LIGHT {mode}", wait_ok=True)[0]
    def rgb(self, r, g, b): return self.cmd(f"RGB {r} {g} {b}", wait_ok=True)[0]
    def home(self): return self.cmd("HOME", wait_ok=True)[0]
    def hold(self): return self.cmd("HOLD ALL", wait_ok=True)[0]
    def scan(self): return self.cmd(f"SCAN S1 {Config.SCAN_MIN} {Config.SCAN_MAX} {Config.SCAN_SPEED}", wait_ok=True)[0]
    def stop_scan(self): return self.cmd("STOPSCAN", wait_ok=True)[0]
    def pose(self, pose):
        a1, a2, a3, a4, a5 = pose
        return self.cmd(f"POSE {a1} {a2} {a3} {a4} {a5} {Config.POSE_TIME_MS}", wait_ok=True)[0]
    def book_hold_on(self): return self.cmd("BOOK_HOLD ON", wait_ok=True)[0]
    def book_hold_off(self): return self.cmd("BOOK_HOLD OFF", wait_ok=True)[0]

    def beep(self, duration_ms=120, freq=2000):
        # ไม่รอ OK กันหน่วง
        return self.cmd(f"BEEP {int(duration_ms)} {int(freq)}", wait_ok=False)[0]

    def poll_shutdown_event(self):
        now = time.time()
        if (now - self._last_poll) < Config.POLL_EVENT_EVERY:
            return None
        self._last_poll = now

        try:
            while self.ser.in_waiting > 0:
                ln = self.ser.readline().decode(errors="ignore").strip()
                if not ln:
                    continue
                if "EVENT SHUTDOWN_BEGIN" in ln:
                    self._seen_shutdown_begin = True
                if "EVENT SHUTDOWN_DONE" in ln:
                    self._seen_shutdown_done = True
        except:
            pass

        if self._seen_shutdown_done:
            return "DONE"
        if self._seen_shutdown_begin:
            return "BEGIN"
        return None

    def close(self):
        try:
            self.light("OFF")
        except:
            pass
        try:
            self.ser.close()
        except:
            pass


class CameraGrabber:
    """ดึงเฟรมจากกล้องในเธรดแยก เพื่อให้ UI FPS สูงขึ้น"""
    def __init__(self):
        self.cam = Picamera2()
        self.need_convert = False
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._th = None

        self._config_camera()

    def _config_camera(self):
        # ตั้งค่ากล้องให้ได้ FPS ดีขึ้น (ไม่ล็อก fps ตายตัว แต่ลดภาระ)
        try:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": Config.CAMERA_FORMAT},
                buffer_count=4
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = False
        except:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": "RGB888"},
                buffer_count=4
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = True

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while self._running:
            try:
                frame = self.cam.capture_array()
                if self.need_convert:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._frame = frame
            except:
                time.sleep(0.003)

    def get_latest(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        try:
            if self._th:
                self._th.join(timeout=0.5)
        except:
            pass
        try:
            self.cam.stop()
        except:
            pass


class AISystem:
    """รัน YOLO ในเธรดแยก (ไม่บล็อก UI) และใช้โมเดลเดียวตรวจทั้งคน+หนังสือ"""
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
        self.book_class_id = self._get_book_class_id()

        self._lock = threading.Lock()
        self._found_person = False
        self._found_book = False
        self._objects = []
        self._last_ai = 0.0
        self._running = False
        self._th = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def _get_book_class_id(self):
        names = self.model.names
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == Config.BOOK_CLASS_NAME.lower():
                    return int(k)
            # fallback
            return int(next(iter(names.keys())))
        return 0

    def push_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame  # เก็บล่าสุดเท่านั้น (ไม่คิวล้น)

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while self._running:
            now = time.time()
            if (now - self._last_ai) < Config.AI_PERIOD_SEC:
                time.sleep(0.002)
                continue

            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.002)
                continue

            self._last_ai = now
            found_person, found_book, objects = self._detect_once(frame)

            with self._lock:
                self._found_person = found_person
                self._found_book = found_book
                self._objects = objects

    def _detect_once(self, frame):
        h, w = frame.shape[:2]
        area_frame = w * h

        # รันครั้งเดียว เอาเฉพาะ class คน + book
        classes = [Config.PERSON_CLASS_ID, self.book_class_id]

        res = self.model(
            frame,
            classes=classes,
            conf=min(Config.CONF_PERSON, Config.CONF_BOOK),  # ให้ต่ำไว้ แล้วคัดทีหลัง
            imgsz=Config.INFER_IMGSZ,
            verbose=False
        )[0]

        objects = []
        found_person = False
        found_book = False

        if res.boxes:
            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                a = ((x2 - x1) * (y2 - y1)) / area_frame

                if cls == Config.PERSON_CLASS_ID:
                    if conf >= Config.CONF_PERSON and a >= Config.MIN_AREA_PERSON:
                        found_person = True
                    label = "Person"
                    # วาดกล่องคนแม้ conf ต่ำก็ได้ตามใจ แต่เอาเฉพาะที่ผ่าน threshold จะสะอาดกว่า
                    if conf >= Config.CONF_PERSON:
                        objects.append((x1, y1, x2, y2, conf, label))

                elif cls == self.book_class_id:
                    if conf >= Config.CONF_BOOK and a >= Config.MIN_AREA_BOOK:
                        found_book = True
                    label = "Book"
                    if conf >= Config.CONF_BOOK:
                        objects.append((x1, y1, x2, y2, conf, label))

        return found_person, found_book, objects

    def get_latest(self):
        with self._lock:
            return self._found_person, self._found_book, list(self._objects)

    def stop(self):
        self._running = False
        try:
            if self._th:
                self._th.join(timeout=0.5)
        except:
            pass


def draw_ui(frame, fps, state, objects, confirm_remaining, armed, no_det_elapsed, shutdown_hint):
    for (x1, y1, x2, y2, conf, label) in objects:
        color = (0, 255, 0) if label == "Person" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"STATE: {state}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"ARMED: {1 if armed else 0}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if confirm_remaining is not None:
        cv2.putText(frame, f"CONFIRM: {confirm_remaining:0.1f}s", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, f"NO-DET: {no_det_elapsed:0.1f}s / {Config.NO_DETECTION_TIMEOUT:.0f}s",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if shutdown_hint:
        cv2.putText(frame, shutdown_hint, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    if Config.SHOW_WINDOW and os.environ.get("DISPLAY", "") == "":
        Config.SHOW_WINDOW = False
        print("[WARN] No DISPLAY found. Disable window mode.")

    robot = RobotArm()

    cam = CameraGrabber()
    cam.start()

    ai = AISystem()
    ai.start()

    state = "WAIT_ARM"
    scan_started = False
    confirm_start = None
    confirm_remaining = None

    book_end = None
    book_last_sec = None
    hold_end = None
    rest_end = None
    sleep_end = None

    last_seen = time.time()
    shutdown_hint = None

    fps = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            now = time.time()

            ev = robot.poll_shutdown_event()
            if ev == "BEGIN":
                shutdown_hint = "UNO: SHUTDOWN BEGIN..."
            elif ev == "DONE":
                print("[INFO] UNO shutdown done -> exit program")
                break

            frame = cam.get_latest()
            if frame is None:
                time.sleep(0.002)
                continue

            # ส่งเฟรมล่าสุดให้ AI เธรด (AI จะเลือกทำตามรอบเวลาเอง)
            ai.push_frame(frame)

            cached_person, cached_book, cached_objects = ai.get_latest()
            if cached_person or cached_book:
                last_seen = now

            armed = robot.is_armed_fast()
            no_det_elapsed = max(0.0, now - last_seen)

            # ---------------- STATES ----------------
            if state == "WAIT_ARM":
                confirm_remaining = None
                scan_started = False
                confirm_start = None
                shutdown_hint = None

                if armed:
                    robot.book_hold_off()
                    robot.stop_scan()
                    robot.light("WHITE")

                    robot.pose(Config.INIT_POSE)
                    wait_pose_done()
                    robot.hold()

                    last_seen = now
                    state = "SCAN"

            elif state == "SCAN":
                confirm_remaining = None

                if not armed:
                    state = "WAIT_ARM"
                else:
                    if not scan_started:
                        last_seen = now
                        robot.book_hold_off()

                        robot.pose(Config.REST_POSE)   # (0,0,0,90,90)
                        wait_pose_done()
                        robot.hold()

                        scan_started = bool(robot.scan())

                    if cached_book:
                        robot.stop_scan()
                        scan_started = False
                        robot.book_hold_off()

                        robot.light(Config.BOOK_LIGHT)
                        robot.pose(Config.BOOK_POSE)
                        wait_pose_done()
                        robot.book_hold_on()

                        state = "BOOK"
                        book_end = now + Config.BOOK_MODE_SEC
                        book_last_sec = None
                        confirm_start = None
                        confirm_remaining = None

                    elif cached_person:
                        if confirm_start is None:
                            confirm_start = now
                        confirm_remaining = max(0.0, Config.CONFIRM_PERSON_SEC - (now - confirm_start))

                        if (now - confirm_start) >= Config.CONFIRM_PERSON_SEC:
                            robot.stop_scan()
                            scan_started = False
                            state = "POSE"
                            confirm_remaining = None
                    else:
                        confirm_start = None

                    if (now - last_seen) > Config.NO_DETECTION_TIMEOUT:
                        robot.stop_scan()
                        scan_started = False
                        robot.book_hold_off()
                        robot.home()
                        robot.light("OFF")
                        state = "SLEEP"
                        sleep_end = now + Config.DEEP_SLEEP_DURATION
                        confirm_start = None
                        confirm_remaining = None

            elif state == "POSE":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    pose = random.choice(Config.POSES)
                    robot.rgb(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    robot.pose(pose)
                    wait_pose_done()
                    robot.hold()

                    hold_end = now + Config.HOLD_POSE_SEC
                    state = "HOLD_POSE"

            elif state == "HOLD_POSE":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if now >= hold_end:
                        robot.cmd("BR 128", wait_ok=True)   # ~50% ของ 255
                        robot.rgb(255, 160, 80)   
                        robot.pose(Config.REST_POSE)
                        wait_pose_done()
                        robot.hold()

                        rest_end = now + Config.REST_SEC
                        state = "REST"

            elif state == "REST":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if cached_book:
                        robot.book_hold_off()
                        robot.light(Config.BOOK_LIGHT)
                        robot.pose(Config.BOOK_POSE)
                        wait_pose_done()
                        robot.book_hold_on()
                        state = "BOOK"
                        book_end = now + Config.BOOK_MODE_SEC
                        book_last_sec = None

                    elif now >= rest_end:
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None

            elif state == "BOOK":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if book_end is not None:
                        remaining = book_end - now

                        # ✅ 5 วินาทีท้าย: บี๊บทุกวินาที (5..1)
                        if 0 < remaining <= 5.0:
                            tick = int(remaining) + 1
                            if book_last_sec != tick:
                                robot.beep(120, 2000)
                                book_last_sec = tick

                    if now >= book_end:
                        robot.book_hold_off()
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None
                        book_last_sec = None

            elif state == "SLEEP":
                confirm_remaining = None
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if now >= sleep_end:
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None

            # ---------------- DISPLAY ----------------
            if Config.SHOW_WINDOW:
                draw_ui(frame, fps, state, cached_objects, confirm_remaining, armed, no_det_elapsed, shutdown_hint)
                cv2.imshow("Controller FAST", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q')):
                    try:
                        robot.stop_scan()
                        robot.book_hold_off()
                        robot.home()
                        robot.light("OFF")
                    except:
                        pass
                    break

            # FPS ของ UI (จาก loop จริง)
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

    finally:
        ai.stop()
        cam.stop()
        if Config.SHOW_WINDOW:
            cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
