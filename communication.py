import serial, time
# Initialize serial connection
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()
while True:
    ser.flush()
    ser.write(b"255,255\n")
    time.sleep(0.1)
ser.close()