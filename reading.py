from gpiozero import Button
import os, time
button = Button(17)
started = False
while True:
        if button.is_pressed:
                if started == False:
                        os.system('python3 ./RoboCup.py')
                else: 
                        os.system('pkill -f "RoboCup.py"')
                        started = False
                time.sleep(1)