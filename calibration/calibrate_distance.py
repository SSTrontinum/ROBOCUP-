import serial, time
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()

#time.sleep(1)
while True:
    ser.write(b"315,315\n")
    input()
    ser.write(b"255,255\n")
    input()
"""
input()
for i in range(140, 255, 10):
    print(f"Testing: {i}")
    ser.write(f"{i+255},{i+255}\n".encode('utf-8'))
    time.sleep(1)
    ser.write(b"255,255\n")
    input()
    ser.write(f"{i+255},{i+255}\n".encode('utf-8'))
    time.sleep(1)
    ser.write(b"255,255\n")
    input()
ser.write(f"{510},{510}\n".encode('utf-8'))
time.sleep(1)
ser.write(b"255,255\n")
input()
ser.write(f"{510},{510}\n".encode('utf-8'))
time.sleep(1)
ser.write(b"255,255\n")
"""

# 255 - 24.5 cm/s