import time
import random
import cv2
import serial
from picamera2 import Picamera2
from ultralytics import YOLO

class Config:
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 115200
    SERIAL_TIMEOUT = 0.1

    FRAME_W = 640
    FRAME_H = 360
    CAMERA_FORMAT = "BGR888"
    SHOW_WINDOW = True

    AI_FRAME_SKIP = 3
    INFER_IMGSZ = 640

    CONF_PERSON = 0.60
    CONF_BOOK = 0.20
    MIN_AREA_PERSON = 0.03
    MIN_AREA_BOOK = 0.01

    MODEL_PERSON_PATH = "yolov8n.pt"
    MODEL_BOOK_PATH = "yolov8n.pt"
    PERSON_CLASS_ID = 0
    BOOK_CLASS_NAME = "book"

    INIT_IDLE_SEC = 0.5
    CONFIRM_PERSON_SEC = 2.0

    NO_DETECTION_TIMEOUT = 10.0
    DEEP_SLEEP_DURATION = 10.0

    BOOK_MODE_SEC = 15.0
    HOLD_POSE_SEC = 2.0
    REST_SEC = 3.0

    INIT_POSE = (0, 0, 0, 0, 0)
    REST_POSE = (0, 0, 0, 0, 0)

    SCAN_MIN = 15
    SCAN_MAX = 165
    SCAN_SPEED = 120  # ✅ เร็วขึ้น

    POSE_TIME_MS = 1200
    POSE_EXTRA_WAIT = 0.25

    POSES = [
        (90, 130, 0, 110, 180),
        (0, 0, 0, 90,  0),
        (74, 0, 52, 0,  180),
        (0, 0, 180, 72, 0),
        (90, 0, 0, 90,  180),
    ]

    BOOK_LIGHT = "WHITE"
    BOOK_POSE = (90, 180, 60, 90, 180)


class RobotArm:
    def __init__(self):
        self.ser = serial.Serial(
            Config.SERIAL_PORT, Config.BAUD_RATE,
            timeout=Config.SERIAL_TIMEOUT, write_timeout=0.2
        )
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _read_lines(self, max_ms=250):
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
                time.sleep(0.003)
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

        lines = self._read_lines(350)
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

    def is_armed(self):
        st = self.status()
        return ("ARMED=1" in st)

    def light(self, mode):
        return self.cmd(f"LIGHT {mode}", wait_ok=True)[0]

    def rgb(self, r, g, b):
        return self.cmd(f"RGB {r} {g} {b}", wait_ok=True)[0]

    def home(self):
        return self.cmd("HOME", wait_ok=True)[0]

    def hold(self):
        return self.cmd("HOLD ALL", wait_ok=True)[0]

    def scan(self):
        # ✅ ถ้า UNO แก้ให้ SCAN ติดแดงเองแล้ว จะไม่จำเป็น
        # แต่ใส่ไว้ก็ไม่เสียหาย เผื่อคุณยังไม่แฟลช UNO ใหม่
        self.light("RED")
        return self.cmd(f"SCAN S1 {Config.SCAN_MIN} {Config.SCAN_MAX} {Config.SCAN_SPEED}", wait_ok=True)[0]

    def stop_scan(self):
        return self.cmd("STOPSCAN", wait_ok=True)[0]

    def pose(self, pose):
        a1, a2, a3, a4, a5 = pose
        return self.cmd(f"POSE {a1} {a2} {a3} {a4} {a5} {Config.POSE_TIME_MS}", wait_ok=True)[0]

    def book_hold_on(self):
        return self.cmd("BOOK_HOLD ON", wait_ok=True)[0]

    def book_hold_off(self):
        return self.cmd("BOOK_HOLD OFF", wait_ok=True)[0]

    def close(self):
        try:
            self.light("OFF")
        except:
            pass
        try:
            self.ser.close()
        except:
            pass


class VisionSystem:
    def __init__(self):
        self.cam = Picamera2()
        self.need_convert = False
        self._config_camera()

        self.model_person = YOLO(Config.MODEL_PERSON_PATH)
        self.model_book = YOLO(Config.MODEL_BOOK_PATH)
        self.book_class_id = self._get_book_class_id()

    def _config_camera(self):
        try:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": Config.CAMERA_FORMAT}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = False
        except:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": "RGB888"}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = True

    def _get_book_class_id(self):
        names = self.model_book.names
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == Config.BOOK_CLASS_NAME.lower():
                    return int(k)
            return int(next(iter(names.keys())))
        return 0

    def get_frame(self):
        frame = self.cam.capture_array()
        if self.need_convert:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def detect(self, frame):
        h, w = frame.shape[:2]
        area_frame = w * h

        res_p = self.model_person(
            frame, classes=[Config.PERSON_CLASS_ID],
            conf=Config.CONF_PERSON, imgsz=Config.INFER_IMGSZ, verbose=False
        )[0]
        res_b = self.model_book(
            frame, classes=[self.book_class_id],
            conf=Config.CONF_BOOK, imgsz=Config.INFER_IMGSZ, verbose=False
        )[0]

        objects = []
        found_person = False
        found_book = False

        if res_p.boxes:
            for box in res_p.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                a = ((x2 - x1) * (y2 - y1)) / area_frame
                if a >= Config.MIN_AREA_PERSON:
                    found_person = True
                objects.append((x1, y1, x2, y2, conf, "Person"))

        if res_b.boxes:
            for box in res_b.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                a = ((x2 - x1) * (y2 - y1)) / area_frame
                if a >= Config.MIN_AREA_BOOK:
                    found_book = True
                objects.append((x1, y1, x2, y2, conf, "Book"))

        return found_person, found_book, objects

    def stop(self):
        try:
            self.cam.stop()
        except:
            pass


def draw_ui(frame, fps, state, objects, confirm_remaining, armed, no_det_elapsed):
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

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    robot = RobotArm()
    vision = VisionSystem()

    state = "WAIT_ARM"
    next_t = time.time() + 0.2

    scan_started = False
    confirm_start = None
    confirm_remaining = None

    book_end = None
    hold_end = None
    rest_end = None
    sleep_end = None

    # ✅ สำคัญ: จะรีเซ็ตทุกครั้งที่เข้า SCAN เพื่อไม่ให้หลับทันที
    last_seen = time.time()

    frame_count = 0
    fps = 0.0
    t0 = time.time()

    cached_person = False
    cached_book = False
    cached_objects = []

    try:
        while True:
            now = time.time()
            frame = vision.get_frame()

            if frame_count % (Config.AI_FRAME_SKIP + 1) == 0:
                cached_person, cached_book, cached_objects = vision.detect(frame)
                if cached_person or cached_book:
                    last_seen = now
            frame_count += 1

            armed = robot.is_armed()
            no_det_elapsed = max(0.0, now - last_seen)

            if state == "WAIT_ARM":
                confirm_remaining = None
                scan_started = False
                confirm_start = None
                if armed:
                    state = "INIT"
                    next_t = now + Config.INIT_IDLE_SEC

            elif state == "INIT":
                confirm_remaining = None
                scan_started = False
                confirm_start = None
                robot.book_hold_off()
                robot.stop_scan()
                robot.light("WHITE")
                robot.pose(Config.INIT_POSE)
                if now >= next_t:
                    # ✅ เข้า SCAN แล้วรีเซ็ต last_seen เพื่อเริ่มนับ 10 วิใหม่
                    last_seen = now
                    state = "SCAN"

            elif state == "SCAN":
                confirm_remaining = None
                if not armed:
                    state = "WAIT_ARM"
                else:
                    # ✅ เข้า SCAN ครั้งแรก: reset timer + สั่ง scan 1 ครั้ง
                    if not scan_started:
                        last_seen = now
                        robot.book_hold_off()
                        scan_started = bool(robot.scan())

                    if cached_book:
                        robot.stop_scan()
                        scan_started = False
                        robot.book_hold_off()
                        robot.light(Config.BOOK_LIGHT)
                        robot.pose(Config.BOOK_POSE)
                        robot.book_hold_on()
                        state = "BOOK"
                        book_end = now + Config.BOOK_MODE_SEC
                        confirm_start = None

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

                    # ✅ หลับเมื่อ "อยู่ใน SCAN" แล้วไม่มี detection ครบ 10 วิจริง ๆ
                    if (now - last_seen) > Config.NO_DETECTION_TIMEOUT:
                        robot.stop_scan()
                        scan_started = False
                        robot.book_hold_off()
                        robot.home()       # 0°
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
                    robot.hold()
                    hold_end = now + Config.HOLD_POSE_SEC
                    state = "HOLD_POSE"

            elif state == "HOLD_POSE":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if now >= hold_end:
                        robot.light("WHITE")
                        robot.pose(Config.REST_POSE)
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
                        robot.book_hold_on()
                        state = "BOOK"
                        book_end = now + Config.BOOK_MODE_SEC
                    elif now >= rest_end:
                        # ✅ กลับไปสแกนต่อจากท่าพัก (ท่าพักคือ 0°)
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None

            elif state == "BOOK":
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if now >= book_end:
                        robot.book_hold_off()
                        # ✅ เข้า SCAN ต่อเลย (ไม่ต้อง INIT)
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None

            elif state == "SLEEP":
                confirm_remaining = None
                if not armed:
                    state = "WAIT_ARM"
                else:
                    if now >= sleep_end:
                        # ✅ ตื่นแล้วเข้า SCAN เลย (ไม่ต้อง INIT)
                        last_seen = now
                        state = "SCAN"
                        scan_started = False
                        confirm_start = None

            if Config.SHOW_WINDOW:
                draw_ui(frame, fps, state, cached_objects, confirm_remaining, armed, no_det_elapsed)
                cv2.imshow("Controller (Rebuild v2)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q')):
                    break

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

    finally:
        vision.stop()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
