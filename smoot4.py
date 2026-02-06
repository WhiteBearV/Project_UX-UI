import time
import random
import cv2
import serial
import threading
from picamera2 import Picamera2
from ultralytics import YOLO

# ===================== CONFIGURATION =====================
class Config:
    # --- Serial Settings ---
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 115200
    TIMEOUT = 0.1
    
    # --- Camera Settings ---
    FRAME_W = 1280
    FRAME_H = 720
    CAMERA_FORMAT = "BGR888" 
    SHOW_WINDOW = True
    
    # --- AI Settings ---
    AI_FRAME_SKIP = 2 
    
    # [แก้ไข] ปรับให้จับ "ง่ายมากๆ"
    CONF_PERSON = 0.60 
    CONF_BOOK = 0.15      # ลดค่าความมั่นใจลง (ยิ่งน้อยยิ่งจับง่าย)
    MIN_AREA_PERSON = 0.03
    MIN_AREA_BOOK = 0.01  # ลดขนาดขั้นต่ำลง (เล่มเล็กก็จับได้)
    
    # --- Models ---
    MODEL_PERSON_PATH = "yolov8n.pt"  # person
    MODEL_BOOK_PATH = "yolov8n.pt"    # book # หรือ best.pt ตามที่คุณมี
    PERSON_CLASS_ID = 0
    BOOK_CLASS_NAME = "book"

    # --- Timings (Seconds) ---
    INIT_IDLE_SEC = 3.0
    CONFIRM_PERSON_SEC = 2.0
    
    # [แก้ไข] เอา BOOK_TRIGGER_SEC ออก (เจอแล้วทำเลย)
    
    # [แก้ไข] เวลา Timeout
    NO_DETECTION_TIMEOUT = 10.0  # ไม่เจออะไร 3 วิ -> กลับบ้าน
    DEEP_SLEEP_DURATION = 10.0  # หลับยาว 10 วิ
    
    BOOK_MODE_SEC = 15.0
    REST_SEC = 5.0

    # --- Rest Pose ---
    # ในโหมด REST ให้กลับมาที่ท่านี้ตลอด
    REST_POSE = (0, 0, 0, 90, 180)
    REST_REASSERT_SEC = 1.0  # ส่งท่าเดิมซ้ำทุกกี่วินาที (กันเซอร์โวไหล)
    # --- Robot Motion Parameters ---
    SCAN_MIN = 15
    SCAN_MAX = 165
    SCAN_SPEED = 25
    POSE_TIME_MS = 1500
    POSE_EXTRA_WAIT = 0.8
    
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
    
    LAMP_NORMAL = (180, 60, 90, 180) # S2-S5

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
            print(f"[INFO] Connected to Robot Arm at {self.port}")
            self.send_command("HOLD ALL", wait=True)
            self.set_light("WHITE")
        except Exception as e:
            print(f"[ERROR] Cannot connect to Serial: {e}")
            self.connected = False

    def _read_response(self, max_ms=100):
        if not self.connected: return []
        end = time.time() + (max_ms / 1000.0)
        lines = []
        while time.time() < end:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line: lines.append(line)
                except: pass
            else:
                time.sleep(0.005)
        return lines

    def send_command(self, cmd, wait=False, retry=1):
        if not self.connected: return False
        cmd = cmd.strip()
        for _ in range(retry + 1):
            try:
                self.ser.write((cmd + "\n").encode())
                self.ser.flush()
                if wait:
                    lines = self._read_response(max_ms=150)
                    if any("OK" in s for s in lines): return True
                    if any("ERR" in s for s in lines):
                        time.sleep(0.05)
                        continue
                else: return True
            except Exception as e:
                print(f"[WARN] Serial Write Error: {e}")
            time.sleep(0.02)
        return False

    def set_light(self, mode):
        return self.send_command(f"LIGHT {mode}", wait=True)
    
    def set_color(self, r, g, b):
        return self.send_command(f"RGB {r} {g} {b}", wait=True)

    def hold_all(self):
        return self.send_command("HOLD ALL", wait=True)

    def start_scan(self, scan_min=None, scan_max=None):
        """เริ่มสแกนฐาน S1 ด้วยช่วงมุมที่กำหนด (ถ้าไม่ระบุจะใช้ค่าใน Config)"""
        self.set_light("RED")
        if scan_min is None: scan_min = Config.SCAN_MIN
        if scan_max is None: scan_max = Config.SCAN_MAX
        return self.send_command(f"SCAN S1 {scan_min} {scan_max} {Config.SCAN_SPEED}", wait=True)

    def stop_scan(self):
        return self.send_command("STOPSCAN", wait=True)
    
    def go_home(self):
        return self.send_command("HOME", wait=True)

    def book_mode_on(self):
        self.stop_scan()
        ln = Config.LAMP_NORMAL
        self.send_command(f"LAMP_NORMAL {ln[0]} {ln[1]} {ln[2]} {ln[3]}", wait=True)
        self.send_command("BOOK_HOLD ON", wait=True)
        self.set_light("WHITE")

    def book_mode_off(self):
        return self.send_command("BOOK_HOLD OFF", wait=True)
    
    def move_pose(self, pose):
        cmd = f"POSE {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {Config.POSE_TIME_MS}"
        return self.send_command(cmd, wait=True)
        
    def close(self):
        if self.connected:
            self.hold_all()
            self.set_light("OFF")
            self.ser.close()

# ===================== VISION SYSTEM =====================
class VisionSystem:
    def __init__(self):
        self.cam = Picamera2()
        self.config_camera()
        
        print("[INFO] Loading AI Models...")
        self.model_person = YOLO(Config.MODEL_PERSON_PATH)
        self.model_book = YOLO(Config.MODEL_BOOK_PATH)
        self.book_class_id = self._get_book_class_id()
        print(f"[INFO] AI Ready. Person ID: {Config.PERSON_CLASS_ID}, Book ID: {self.book_class_id}")

    def config_camera(self):
        try:
            cfg = self.cam.create_video_configuration(
                main={"size": (Config.FRAME_W, Config.FRAME_H), "format": Config.CAMERA_FORMAT}
            )
            self.cam.configure(cfg)
            self.cam.start()
            self.need_convert = False 
        except Exception as e:
            print(f"[WARN] Format error: {e}. Fallback to RGB888")
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
        frame_area = w * h
        
        # ใช้ conf ต่ำๆ เพื่อให้จับง่าย
        res_p = self.model_person(frame, classes=[Config.PERSON_CLASS_ID], conf=Config.CONF_PERSON, verbose=False)[0]
        res_b = self.model_book(frame, classes=[self.book_class_id], conf=Config.CONF_BOOK, verbose=False)[0]
        
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
        self.cam.stop()

# ===================== MAIN CONTROLLER =====================
def draw_ui(frame, fps, state, objects):
    for (x1, y1, x2, y2, conf, label) in objects:
        color = (0, 255, 0) if label == "Person" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, Config.FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

def main():
    robot = RobotArm(Config.SERIAL_PORT, Config.BAUD_RATE)
    vision = VisionSystem()
    
    state = "INIT"
    next_state_time = time.time() + Config.INIT_IDLE_SEC
    
    confirm_start_time = None
    book_end_time = None

    # สแกนแบบ "ล็อกด้าน" ตอนเจอคนแต่ยังไม่ยืนยัน
    current_scan_min = None
    current_scan_max = None
    last_person_side = None  # "L" หรือ "R"
    rest_last_cmd_time = 0.0

    
    # [แก้ไข] ตัวแปรจับเวลาเมื่อไม่เจออะไร
    last_seen_time = time.time()
    wake_up_time = 0
    
    last_pose_idx = -1
    scan_active = False
    
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
            
            # --- AI Logic ---
            if frame_count % (Config.AI_FRAME_SKIP + 1) == 0:
                cached_person, cached_book, cached_objects = vision.detect(frame)
                
                # [แก้ไข] อัปเดตเวลาล่าสุดที่เจอสิ่งมีชีวิตหรือหนังสือ
                if cached_person or cached_book:
                    last_seen_time = now
            frame_count += 1
            
            # --- State Machine ---
            if state == "INIT":
                if now >= next_state_time:
                    state = "SCAN"
                    scan_active = False
                    rest_last_cmd_time = 0.0

            elif state == "SCAN":
                # 1) เจอหนังสือ -> เข้า BOOK ทันที
                if cached_book:
                    robot.book_mode_on()
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                    scan_active = False
                    confirm_start_time = None
                    current_scan_min, current_scan_max = None, None
                    last_person_side = None
                
                # 2) Timeout: ไม่เจออะไรนาน -> หลับ
                elif (now - last_seen_time) > Config.NO_DETECTION_TIMEOUT:
                    print("[INFO] No detection. Going to sleep...")
                    robot.stop_scan()
                    robot.go_home()
                    robot.set_light("OFF")
                    scan_active = False
                    state = "DEEP_SLEEP"
                    wake_up_time = now + Config.DEEP_SLEEP_DURATION
                    confirm_start_time = None
                    current_scan_min, current_scan_max = None, None
                    last_person_side = None
                
                # 3) โหมดปกติ: สแกนหา "คน"
                else:
                    # หา "ด้าน" ของคนจากกรอบตรวจจับ (ซ้าย/ขวา)
                    person_side = None
                    if cached_person and cached_objects:
                        # เอากล่อง Person ที่ใหญ่สุดเป็นตัวอ้างอิง
                        person_boxes = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, label) in cached_objects if label == "Person"]
                        if person_boxes:
                            # เลือกกล่องที่มีพื้นที่มากสุด
                            (x1, y1, x2, y2, conf) = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                            cx = (x1 + x2) * 0.5
                            person_side = "L" if cx < (Config.FRAME_W * 0.5) else "R"
                
                    # ถ้าไม่เจอคน -> สแกนเต็มช่วงปกติ
                    if not cached_person:
                        confirm_start_time = None
                        last_person_side = None
                        desired_min, desired_max = Config.SCAN_MIN, Config.SCAN_MAX
                    else:
                        # เจอคนแต่ยังไม่ยืนยัน: "ล็อกสแกน" ไปด้านที่เจอ เพื่อไม่ให้หันหนีไปอีกด้าน
                        last_person_side = person_side or last_person_side or "L"
                        mid = int((Config.SCAN_MIN + Config.SCAN_MAX) * 0.5)
                        if last_person_side == "L":
                            desired_min, desired_max = Config.SCAN_MIN, mid
                        else:
                            desired_min, desired_max = mid, Config.SCAN_MAX
                
                        # เริ่ม/นับเวลายืนยันคน (ต้องเห็นต่อเนื่อง)
                        if confirm_start_time is None:
                            confirm_start_time = now
                
                    # เริ่มสแกน หรือรีสตาร์ทสแกนเมื่อช่วงมุมเปลี่ยน
                    if (not scan_active) or (current_scan_min != desired_min) or (current_scan_max != desired_max):
                        robot.start_scan(desired_min, desired_max)
                        scan_active = True
                        current_scan_min, current_scan_max = desired_min, desired_max
                
                    # ถ้าเห็นคนต่อเนื่องครบเวลา -> ค่อยหยุดสแกนแล้วเข้าสู่ CONFIRM (เพื่อทำท่า)
                    if cached_person and confirm_start_time is not None:
                        if (now - confirm_start_time) >= Config.CONFIRM_PERSON_SEC:
                            robot.stop_scan()
                            scan_active = False
                            state = "CONFIRM"
                            # ไม่รีเซ็ต confirm_start_time เพื่อให้ CONFIRM ผ่านทันที
                
                
            elif state == "DEEP_SLEEP":
                # นอนรอจนครบ 10 วินาที
                if now >= wake_up_time:
                    print("[INFO] Waking up!")
                    last_seen_time = now # รีเซ็ตเวลาเพื่อไม่ให้หลับซ้ำทันที
                    state = "SCAN"
                    scan_active = False

            elif state == "CONFIRM":
                if cached_book:
                    robot.book_mode_on()
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                elif not cached_person:
                    state = "SCAN"
                    scan_active = False
                else:
                    if (now - confirm_start_time) >= Config.CONFIRM_PERSON_SEC:
                        # สุ่มท่า + สุ่มสี
                        idx = random.randrange(len(Config.POSES))
                        if len(Config.POSES) > 1:
                            while idx == last_pose_idx: idx = random.randrange(len(Config.POSES))
                        last_pose_idx = idx
                        
                        r = random.randint(0, 255)
                        g = random.randint(0, 255)
                        b = random.randint(0, 255)
                        robot.set_color(r, g, b) 
                        
                        robot.move_pose(Config.POSES[idx])
                        state = "POSE"
                        next_state_time = now + (Config.POSE_TIME_MS / 1000.0) + Config.POSE_EXTRA_WAIT

            elif state == "POSE":
                if now >= next_state_time:
                    if cached_book:
                        robot.book_mode_on()
                        state = "BOOK"
                        book_end_time = now + Config.BOOK_MODE_SEC
                    else:
                        robot.hold_all()
                        robot.set_color(20, 20, 20) 
                        state = "REST"
                        rest_last_cmd_time = 0.0
                        next_state_time = now + Config.REST_SEC

            elif state == "REST":
                # กลับมาท่า REST ตลอด (0,0,0,90,180)
                if rest_last_cmd_time == 0.0:
                    robot.move_pose(Config.REST_POSE)
                    rest_last_cmd_time = now
                elif (now - rest_last_cmd_time) >= Config.REST_REASSERT_SEC:
                    robot.move_pose(Config.REST_POSE)
                    rest_last_cmd_time = now
                
                if cached_book:
                    robot.book_mode_on()
                    state = "BOOK"
                    book_end_time = now + Config.BOOK_MODE_SEC
                    rest_last_cmd_time = 0.0
                elif now >= next_state_time:
                    rest_last_cmd_time = 0.0
                    last_seen_time = now
                    state = "SCAN"
                    scan_active = False
                
                
            elif state == "BOOK":
                if now >= book_end_time:
                    robot.book_mode_off()
                    robot.hold_all()
                    last_seen_time = now
                    state = "SCAN"
                    scan_active = False

            # --- Display ---
            if Config.SHOW_WINDOW:
                draw_ui(frame, fps, state, cached_objects)
                cv2.imshow("Robot Arm Controller", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    robot.stop_scan()
                    robot.book_mode_off()
                    robot.go_home()
                    time.sleep(1.5)
                    break
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt)

    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down...")
        vision.stop()
        cv2.destroyAllWindows()
        robot.close()

if __name__ == "__main__":
    main()