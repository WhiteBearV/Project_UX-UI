import time
import serial

PORT = "/dev/ttyACM0"   # ปรับถ้าของนายไม่ใช่
BAUD = 115200

def send(ser, s):
    ser.write((s + "\n").encode())
    ser.flush()
    # อ่านตอบกลับ (ถ้ามี)
    time.sleep(0.02)
    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(2)  # รอ UNO รีเซ็ตหลังเปิดพอร์ต
    send(ser, "CLEAR")
    send(ser, "BR 120")
    send(ser, "SET 0 255 0 0")
    send(ser, "SET 1 0 255 0")
    send(ser, "SET 2 0 0 255")
    time.sleep(1)

    send(ser, "FILL 255 0 0")
    time.sleep(1)
    send(ser, "FILL 0 255 0")
    time.sleep(1)
    send(ser, "CLEAR")

    # rainbow + rotate demo
    for _ in range(200):
        send(ser, "RAINBOW 5")
