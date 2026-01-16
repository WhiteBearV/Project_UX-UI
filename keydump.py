import sys, tty, termios

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setraw(fd)
    print("กดปุ่ม (กด q เพื่อออก) — จะแสดง byte ที่ได้รับ")
    while True:
        b = sys.stdin.buffer.read(1)
        print("byte:", b, "hex:", b.hex())
        if b == b"q":
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
