#!/usr/bin/python

# MIT License
# 
# Copyright (c) 2017 John Bryan Moore
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import VL53L0X

# Create a VL53L0X object for device on PCA9548A bus 0
tof1 = VL53L0X.VL53L0X(tca9548a_num=0, tca9548a_addr=0x70)
# Create a VL53L0X object for device on PCA9548A bus 1
tof2 = VL53L0X.VL53L0X(tca9548a_num=1, tca9548a_addr=0x70)
# Create a VL53L0X object for device on PCA9548A bus 2
tof3 = VL53L0X.VL53L0X(tca9548a_num=2, tca9548a_addr=0x70)

tof1.open()
tof2.open()
tof3.open()

# Start ranging on PCA9548A bus 0
tof1.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
# Start ranging on PCA9548A bus 1
tof2.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
# Start ranging on PCA9548A bus 2
tof3.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)

timing = tof1.get_timing()
if timing < 20000:
    timing = 20000
print("Timing %d ms" % (timing/1000))
front_distances = []
right_distances = []
left_distances = []
ferror, rerror, lerror = -1, 53, 51
try:
    for count in range(1, 100):
        # Get distance from VL53L0X on PCA9548A bus 0
        distance = int(tof1.get_distance()) - ferror
        front_distances.append(distance)
        print("Front: %d mm, %d cm, %d" % (distance, (distance/10), count))

        # Get distance from VL53L0X on PCA9548A bus 1
        distance = int(tof2.get_distance()) - rerror
        right_distances.append(distance)
        print("Right: %d mm, %d cm, %d" % (distance, (distance/10), count))
        
        # Get distance from VL53L0X on PCA9548A bus 2
        distance = int(tof3.get_distance()) - lerror
        left_distances.append(distance)
        print("Left: %d mm, %d cm, %d" % (distance, (distance/10), count))
        print("-----------------------------")

        time.sleep(0.1)

    tof1.stop_ranging()
    tof2.stop_ranging()
    tof3.stop_ranging()

    tof1.close()
    tof2.close()
    tof3.close()
except Exception as e:
    print(f"Error: {e}")
    exit()
print(f"Front Average = {sum(front_distances)/len(front_distances)}")
print(f"Right Average = {sum(right_distances)/len(right_distances)}")
print(f"Left Average = {sum(left_distances)/len(left_distances)}")
exit()
