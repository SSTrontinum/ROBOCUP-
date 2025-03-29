import serial, time
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()
ser.write(b"0,510\n")
#time.sleep(1)
#ser.write(b"255,255\n")