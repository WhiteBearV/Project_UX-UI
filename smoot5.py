\
import time
import random
import cv2
import serial
from picamera2 import Picamera2
from ultralytics import YOLO

# ===================== CONFIGURATION =====================
class Config:
    # --- Serial Settings ---
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 115200
    TIMEOUT = 0.1

    # --- Camera Settings ---
    # เพิ่ม FPS: ลดความละเอียดลง (แนะนำ 640x360 หรือ 640x480)
    FRAME_W = 640
    FRAME_H = 360
    CAMERA_FORMAT = "BGR888"
    SHOW_WINDOW = True

    # --- AI Settings ---
    # เพิ่ม FPS: ให้ AI ทำงานห่างขึ้น
    AI_FRAME_SKIP = 4
    # ขนาดภาพตอน infer (เล็กลง = เร็วขึ้น)
    INFER_IMGSZ = 640

    CONF_PERSON = 0.60
    CONF_BOOK = 0.15
    MIN_AREA_PERSON = 0.03
    MIN_AREA_BOOK = 0.01

    # --- Models ---
    MODEL_PERSON_PATH = "yolov8n.pt"
    MODEL_BOOK_PATH = "yolov8n.pt"
    PERSON_CLASS_ID = 0
    BOOK_CLASS_NAME = "book"

    # --- Timings (Seconds) ---
    INIT_IDLE_SEC = 3.0
    CONFIRM_PERSON_SEC = 2.0

    NO_DETECTION_TIMEOUT = 10.0
    DEEP_SLEEP_DURATION = 10.0

    BOOK_MODE_SEC = 15.0
    REST_SEC = 5.0

    # --- Rest Pose ---
    REST_POSE = (90, 0, 0, 90, 180)
    REST_REASSERT_SEC = 1.0

    # --- Robot Motion Parameters ---
    SCAN_MIN = 15
    SCAN_MAX = 165
    SCAN_SPEED = 25
    POSE_TIME_MS = 1000        # ลดเวลาท่า ให้ไม่อืด
    POSE_EXTRA_WAIT = 0.25    # ลดเวลารอ

    # --- Poses (S1, S2, S3, S4, S5) ---
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

    # ท่าหนังสือสำหรับ S2-S5 (ต้องสั่ง POSE ก่อน แล้วค่อย BOOK_HOLD ON)
    LAMP_NORMAL = (180, 60, 90, 90)   # S2,S3,S4,S5
    BOOK_POSE_SETTLE_SEC = 1.2         # รอให้ถึงท่าแล้วค่อย lock


# ===================== HARDWARE INTERFACE =====================
class RobotArm:
    def __init__(self, port, baud):
        self.ser = None
        self.port = port
        self.baud = baud
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=Config.TIMEOUT, write_timeout=0.2)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.connected = True
            print(f"[INFO] Connected to UNO at {self.port}")
            # ไม่สั่งเปิดไฟตอนเริ่มรัน
        except Exception as e:
            print(f"[ERROR] Cannot connect to Serial: {e}")
            self.connected = False

    def _read_response(self, max_ms=160):
        if not self.connected:
            return []
        end = time.time() + (max_ms / 1000.0)
        lines = []
        while time.time() < end:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        lines.append(line)
                except:
                    pass
            else:
                time.sleep(0.005)
        return lines

    def send_command(self, cmd, wait=False, retry=0):
        """ถ้า UNO ยังไม่ armed จะตอบ ERR_NOT_ARMED -> คืน False (Pi จะไม่พยายามซ้ำ)"""
        if not self.connected:
            return False

        cmd = cmd.strip()
        for _ in range(retry + 1):
            try:
                self.ser.write((cmd + "\n").encode())
                self.ser.flush()
                if wait:
                    lines = self._read_response(max_ms=220)
                    if any("ERR_NOT_ARMED" in s for s in lines):
                        return False
                    if any("OK" in s for s in lines):
                        return True
                    if any("ERR" in s for s in lines):
                        return False
                else:
                    return True
            except Exception as e:
                print(f"[WARN] Serial Write Error: {e}")
            time.sleep(0.01)
        return False

    # คำสั่งด้านล่าง: ถ้า UNO ยังไม่ armed จะเงียบ (False) และ Pi จะไม่ spam
    def set_light(self, mode):
        return self.send_command(f"LIGHT {mode}", wait=True)

    def set_color(self, r, g, b):
        return self.send_command(f"RGB {r} {g} {b}", wait=True)

    def hold_all(self):
        return self.send_command("HOLD ALL", wait=True)

    def start_scan(self, scan_min=None, scan_max=None):
        # ถ้ายังไม่ armed จะไม่ทำอะไร
        if scan_min is None:
            scan_min = Config.SCAN_MIN
        if scan_max is None:
            scan_max = Config.SCAN_MAX
        # เปิดไฟตอนสแกน เฉพาะเมื่อ UNO armed แล้ว (ถ้าไม่ armed คำสั่งจะถูก reject)
        self.set_light("RED")
        return self.send_command(f"SCAN S1 {scan_min} {scan_max} {Config.SCAN_SPEED}", wait=True)

    def stop_scan(self):
        return self.send_command("STOPSCAN", wait=True)

    def go_home(self):
        return self.send_command("HOME", wait=True)

    def rest_pose(self):
        return self.send_command("REST", wait=True)

    def move_pose(self, pose):
        cmd = f"POSE {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {Config.POSE_TIME_MS}"
        return self.send_command(cmd, wait=True)

    def book_mode_on(self, last_base_angle=90):
        # ไปท่าหนังสือก่อน แล้วค่อย lock
        self.stop_scan()
        ln = Config.LAMP_NORMAL
        pose = (int(last_base_angle), int(ln[0]), int(ln[1]), int(ln[2]), int(ln[3]))
        ok = self.move_pose(pose)
        if ok:
            time.sleep(Config.BOOK_POSE_SETTLE_SEC)
            self.send_command("BOOK_HOLD ON", wait=True)
            self.set_light("WHITE")

    def book_mode_off(self):
        return self.send_command("BOOK_HOLD OFF", wait=True)

    def close(self):
        if self.connected:
            try:
                self.send_command("LIGHT OFF", wait=True)
                self.ser.close()
            except:
                pass


# ===================== VISION SYSTEM =====================
class VisionSystem:
    def __init__(self):
        self.cam = Picamera2()
        self._config_camera()

        print("[INFO] Loading AI Models...")
        self.model_person = YOLO(Config.MODEL_PERSON_PATH)
        self.model_book = YOLO(Config.MODEL_BOOK_PATH)
        self.book_class_id = self._get_book_class_id()
        print(f"[INFO] AI Ready. Person ID: {Config.PERSON_CLASS_ID}, Book ID: {self.book_class_id}")

    def _config_camera(self):
        try:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": Config.CAMERA_FORMAT}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = False
            print("[INFO] Camera format: BGR888")
        except Exception as e:
            print(f"[WARN] Format error: {e}. Fallback to RGB888")
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": "RGB888"}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = True
            print("[INFO] Camera format: RGB888 (will convert to BGR)")

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
        frame_area = w * h

        res_p = self.model_person(
            frame, classes=[Config.PERSON_CLASS_ID], conf=Config.CONF_PERSON,
            verbose=False, imgsz=Config.INFER_IMGSZ
        )[0]
        res_b = self.model_book(
            frame, classes=[self.book_class_id], conf=Config.CONF_BOOK,
            verbose=False, imgsz=Config.INFER_IMGSZ
        )[0]

        detected_objects = []
        found_person = False
        found_book = False

        if res_p.boxes:
            for box in res_p.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = ((x2-x1)*(y2-y1)) / frame_area
                if area >= Config.MIN_AREA_PERSON:
                    found_person = True
                detected_objects.append((x1, y1, x2, y2, conf, "Person"))

        if res_b.boxes:
            for box in res_b.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = ((x2-x1)*(y2-y1)) / frame_area
                if area >= Config.MIN_AREA_BOOK:
                    found_book = True
                detected_objects.append((x1, y1, x2, y2, conf, "Book"))

        return found_person, found_book, detected_objects

    def stop(self):
        try:
            self.cam.stop()
        except:
            pass


def draw_ui(frame, fps, state, objects):
    for (x1, y1, x2, y2, conf, label) in objects:
        color = (0, 255, 0) if label == "Person" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"STATE: {state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, Config.FRAME_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    robot = RobotArm(Config.SERIAL_PORT, Config.BAUD_RATE)
    vision = VisionSystem()

    state = "INIT"
    next_state_time = time.time() + Config.INIT_IDLE_SEC
    book_end_time = None

    current_scan_min = None
    current_scan_max = None
    last_person_side = None
    confirm_start_time = None

    rest_last_cmd_time = 0.0
    last_seen_time = time.time()
    wake_up_time = 0.0

    last_pose_idx = -1
    scan_active = False

    frame_count = 0
    fps = 0.0
    t0 = time.time()

    cached_person = False
    cached_book = False
    cached_objects = []

    last_base_angle = 90.0

    try:
        while True:
            now = time.time()
            frame = vision.get_frame()

            if frame_count % (Config.AI_FRAME_SKIP + 1) == 0:
                cached_person, cached_book, cached_objects = vision.detect(frame)
                if cached_person or cached_book:
                    last_seen_time = now
            frame_count += 1

            if state == "INIT":
                if now >= next_state_time:
                    state = "SCAN"
                    scan_active = False
                    confirm_start_time = None
                    last_person_side = None
                    current_scan_min, current_scan_max = None, None
                    rest_last_cmd_time = 0.0

            elif state == "SCAN":
                if cached_book:
                    robot.book_mode_on(last_base_angle=last_base_angle)
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                    scan_active = False
                    confirm_start_time = None
                    current_scan_min, current_scan_max = None, None
                    last_person_side = None

                elif (now - last_seen_time) > Config.NO_DETECTION_TIMEOUT:
                    robot.stop_scan()
                    robot.go_home()
                    robot.set_light("OFF")
                    scan_active = False
                    state = "DEEP_SLEEP"
                    wake_up_time = now + Config.DEEP_SLEEP_DURATION
                    confirm_start_time = None
                    current_scan_min, current_scan_max = None, None
                    last_person_side = None

                else:
                    person_side = None
                    if cached_person and cached_objects:
                        person_boxes = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, label) in cached_objects if label == "Person"]
                        if person_boxes:
                            (x1, y1, x2, y2, conf) = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                            cx = (x1 + x2) * 0.5
                            person_side = "L" if cx < (Config.FRAME_W * 0.5) else "R"

                    if not cached_person:
                        confirm_start_time = None
                        last_person_side = None
                        desired_min, desired_max = Config.SCAN_MIN, Config.SCAN_MAX
                        last_base_angle = (Config.SCAN_MIN + Config.SCAN_MAX) * 0.5
                    else:
                        last_person_side = person_side or last_person_side or "L"
                        mid = int((Config.SCAN_MIN + Config.SCAN_MAX) * 0.5)
                        if last_person_side == "L":
                            desired_min, desired_max = Config.SCAN_MIN, mid
                            last_base_angle = (Config.SCAN_MIN + mid) * 0.5
                        else:
                            desired_min, desired_max = mid, Config.SCAN_MAX
                            last_base_angle = (mid + Config.SCAN_MAX) * 0.5

                        if confirm_start_time is None:
                            confirm_start_time = now

                    if (not scan_active) or (current_scan_min != desired_min) or (current_scan_max != desired_max):
                        ok = robot.start_scan(desired_min, desired_max)
                        # ถ้า UNO ยังไม่ armed -> ok=False (Pi จะไม่ spam)
                        scan_active = bool(ok)
                        current_scan_min, current_scan_max = desired_min, desired_max

                    if cached_person and confirm_start_time is not None:
                        if (now - confirm_start_time) >= Config.CONFIRM_PERSON_SEC:
                            robot.stop_scan()
                            scan_active = False
                            state = "CONFIRM"

            elif state == "DEEP_SLEEP":
                if now >= wake_up_time:
                    last_seen_time = now
                    state = "SCAN"
                    scan_active = False
                    confirm_start_time = None

            elif state == "CONFIRM":
                if cached_book:
                    robot.book_mode_on(last_base_angle=last_base_angle)
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                elif not cached_person:
                    state = "SCAN"
                    scan_active = False
                    confirm_start_time = None
                else:
                    idx = random.randrange(len(Config.POSES))
                    if len(Config.POSES) > 1:
                        while idx == last_pose_idx:
                            idx = random.randrange(len(Config.POSES))
                    last_pose_idx = idx

                    robot.set_color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    robot.move_pose(Config.POSES[idx])

                    state = "POSE"
                    next_state_time = now + (Config.POSE_TIME_MS / 1000.0) + Config.POSE_EXTRA_WAIT
                    confirm_start_time = None

            elif state == "POSE":
                if now >= next_state_time:
                    if cached_book:
                        robot.book_mode_on(last_base_angle=last_base_angle)
                        state = "BOOK"
                        book_end_time = now + Config.BOOK_MODE_SEC
                    else:
                        robot.hold_all()
                        robot.set_color(20, 20, 20)
                        state = "REST"
                        rest_last_cmd_time = 0.0
                        next_state_time = now + Config.REST_SEC

            elif state == "REST":
                if rest_last_cmd_time == 0.0:
                    robot.rest_pose()
                    rest_last_cmd_time = now
                elif (now - rest_last_cmd_time) >= Config.REST_REASSERT_SEC:
                    robot.rest_pose()
                    rest_last_cmd_time = now

                if cached_book:
                    robot.book_mode_on(last_base_angle=last_base_angle)
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                    rest_last_cmd_time = 0.0
                elif now >= next_state_time:
                    rest_last_cmd_time = 0.0
                    last_seen_time = now
                    state = "SCAN"
                    scan_active = False
                    confirm_start_time = None

            elif state == "BOOK":
                if now >= book_end_time:
                    robot.book_mode_off()
                    robot.hold_all()
                    last_seen_time = now
                    state = "SCAN"
                    scan_active = False
                    confirm_start_time = None

            if Config.SHOW_WINDOW:
                draw_ui(frame, fps, state, cached_objects)
                cv2.imshow("Robot Arm Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    robot.stop_scan()
                    robot.book_mode_off()
                    robot.go_home()
                    time.sleep(0.5)
                    break

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
