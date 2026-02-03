import time
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

FRAME_W, FRAME_H = 1280, 720
CONF = 0.35
SHOW = True   # ถ้า SSH ไม่มีจอ ให้เปลี่ยนเป็น False

# COCO class ids: person=0, book=73
TARGET = {0: "person", 73: "book"}

def main():
    # ---- camera ----
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

    # ---- yolo ----
    model = YOLO("yolo11n.pt")  # ถ้ามีไฟล์ .pt อยู่แล้วใส่ชื่อไฟล์คุณแทนได้
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            rgb = cam.capture_array()                 # RGB888
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # YOLO inference
            res = model(frame, classes=[0, 73], conf=0.25, verbose=False)[0]

            # draw only person+book
            for b in res.boxes:
                cls = int(b.cls[0])
                if cls not in TARGET:
                    continue
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{TARGET[cls]} {conf:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, FRAME_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if SHOW:
                cv2.imshow("YOLO PiCam person+book", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
            else:
                # headless: เซฟรูปทุก 30 เฟรม (ประมาณ 1 วินาทีที่ 30fps)
                if int(time.time()) % 2 == 0:
                    cv2.imwrite("yolo_dbg.jpg", frame)

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
