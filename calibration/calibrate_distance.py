
import serial, time
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()

#time.sleep(1)

"""
input()
for i in range(10, 255, 10):
    print(f"Testing: {i}")
    ser.write(f"{i+255},{i+255}\n".encode('utf-8'))
    time.sleep(1)
    ser.write(b"255,255\n")
    input()
    ser.write(f"{i+255},{i+255}\n".encode('utf-8'))
    time.sleep(1)
    ser.write(b"255,255\n")
    input()
"""
ser.write(f"{255-100},{255+100}\n".encode('utf-8'))
input()
ser.write(b"255,255\n")
input()
ser.write(f"{255+170},{255+170}\n".encode('utf-8'))
time.sleep(1)
ser.write(b"255,255\n")


# 255 - 24.5 cm/s
