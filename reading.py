from gpiozero import Button
import os, time, serial
button = Button(17)
while True:
        if button.is_pressed:
                print("starting")
                ser = serial.Serial('/dev/serial0', 9600, timeout=1)
                ser.reset_input_buffer()
                ser.flush()
                os.system('python3 RoboCup.py')
                time.sleep(1)
