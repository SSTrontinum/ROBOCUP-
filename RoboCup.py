###############
### IMPORTS ###
###############
import cv2, serial, time, math, sys
import VL53L0X, py_qmc5883l
from gpiozero import Button
import numpy as np
from libcamera import Transform, controls
from picamera2 import Picamera2, Preview
import board, busio

####f###############
### CALIBRATION ###
###################
GREEN_LOWER_THRESHOLD = (45, 100, 75)
GREEN_UPPER_THRESHOLD = (80, 255, 255)
RED_LOWER_THRESHOLD_1 = (0, 65, 75)
RED_UPPER_THRESHOLD_1 = (5, 255, 255)
RED_LOWER_THRESHOLD_2 = (175, 65, 75)
RED_UPPER_THRESHOLD_2 = (180, 255, 255)
F_ERROR, R_ERROR, L_ERROR = -1, 53, 51
GYRO_CALIBRATION = [[1.1734590532408868, 0.04732512632887426, 2214.3642882299787], [0.04732512632887431, 1.012911794110473, 4101.536882585206], [0.0, 0.0, 1.0]]
GYRO_DECLINATION = 0.19
ROBOT_WIDTH = 11.4
ROBOT_LENGTH = 15.0
OBSTACLE_DISTANCE_SPOTTED_THRESHOLD = 50
OBSTACLE_DETECTION_THRESHOLD = 30
INNER = 55
BASE = 50
MAX_F_DIST = 200
MAX_S_DIST = 300
# wheel diameter = 53.25 mm
# robot width = 114.00 mm

########################
### GLOBAL VARIABLES ###
########################
# PID config
kp = 20
ki = 0 # set to 0
kd = 5 # should be 10 to 50 times larger than kp
error_sum = 0
prev_error = 0
kr = 0.15
prev_time = time.time()
started = True

######################
### INITIALISATION ###
######################
## SERIAL ##
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()

## CAMERA ##
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration({"size": (258, 194)}, transform=Transform(hflip=True, vflip=True))
picam2.configure(preview_config)

if '-n' in sys.argv:
    picam2.start(Preview.QTGL)
else:
    picam2.start(show_preview=True)

## TOF SENSORS ###
F_SENSOR = VL53L0X.VL53L0X(tca9548a_num=0, tca9548a_addr=0x70)
R_SENSOR = VL53L0X.VL53L0X(tca9548a_num=1, tca9548a_addr=0x70)
L_SENSOR = VL53L0X.VL53L0X(tca9548a_num=2, tca9548a_addr=0x70)
F_SENSOR.open()
R_SENSOR.open()
L_SENSOR.open()
F_SENSOR.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
R_SENSOR.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
L_SENSOR.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)

## GYRO ##
GYRO = py_qmc5883l.QMC5883L()
GYRO.declination = GYRO_DECLINATION
GYRO.calibration = GYRO_CALIBRATION

## Button ##
button = Button(27)

#################
### FUNCTIONS ###
#################
def CMS2AN(cms_speed):
    return 0.000296378 * cms_speed**5 - 0.0156624 * cms_speed**4 + 0.303438 * cms_speed**3 - 2.3577 * cms_speed**2 + 8.83765 * cms_speed + 19.42041

def I2C():
    try:
        data = [0, 0, 0, 0]
        data[0] = min(MAX_F_DIST, F_SENSOR.get_distance() - F_ERROR)
        data[1] = min(MAX_S_DIST, R_SENSOR.get_distance() - R_ERROR)
        data[2] = min(MAX_S_DIST, L_SENSOR.get_distance() - L_ERROR)
        data[3] = GYRO.get_bearing()
        new_f = F_SENSOR.get_distance() - F_ERROR
        if abs(new_f - data[0]) > 10:
            data[0] = -1 # Invalid
        else:
            data[0] = (new_f + data[0]) / 2
        return data
    except: return [MAX_F_DIST, MAX_S_DIST, MAX_S_DIST, 90]

def move_until_detect(side):
    if side == 'r':
        current = I2C()[1]
        if current < OBSTACLE_DETECTION_THRESHOLD:
            ser.write(b"205,205\n")
            while abs(I2C()[1] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
            ser.write(b"255,255\n")
        else:
            ser.write(b"305,305\n")
            while abs(I2C()[1] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
            ser.write(b"255,255\n")
    else:
        current = I2C()[2]
        if current < OBSTACLE_DETECTION_THRESHOLD:
            ser.write(b"205,205\n")
            while abs(I2C()[2] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
            ser.write(b"255,255\n")
        else:
            ser.write(b"305,305\n")
            while abs(I2C()[2] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
            ser.write(b"255,255\n")
    move(5, 1)

def move_until_undetect(side):
    start_time = time.time()
    if side == 'r':
        current = I2C()[1]
        ser.write(b"305,305\n")
        while abs(I2C()[1] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
        ser.write(b"255,255\n")
    else:
        current = I2C()[2]
        ser.write(b"305,305\n")
        while abs(I2C()[2] - current) < OBSTACLE_DISTANCE_SPOTTED_THRESHOLD: pass
        ser.write(b"255,255\n")
    return time.time() - start_time

def move_speed(speed, t):
    ser.write(f"{speed},{speed}\n".encode('utf-8'))
    time.sleep(t)
    ser.write(b"255,255\n")

def check_for_line():
    image = picam2.capture_array()
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    removed = int(kr * imagetemp.shape[0])
    img = imagetemp[int(kr * imagetemp.shape[0]):]
    rows, cols, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bcx, bcy = 0, 0
    ret, btemplate = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_OPEN, kernel)
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_CLOSE, kernel)
    ret, btemplate = cv2.threshold(btemplate, 128, 255, cv2.THRESH_BINARY_INV)
    for i in range(rows):
        for j in range(cols):
            if i <= 5 or i >= (rows - 5):
                btemplate[i][j] = 255
            if j <= 5 or j >= (cols - 5):
                btemplate[i][j] = 255
    contours, hierarchy = cv2.findContours(btemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index in range(len(contours)):
        cnt = contours[index]
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 30000:
            return True
    return False

def cv2_imshow(winname, img, x, y):
    if '-n' in sys.argv: return
    cv2.namedWindow(winname)  
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, img)

def turn(direction, angle):
    direction = direction.upper()
    if direction == "L":
        expected_bearing = I2C()[3] + angle
    else:
        expected_bearing = I2C()[3] - angle
    if expected_bearing < 0:
        expected_bearing += 360
    elif expected_bearing > 360:
        expected_bearing -= 360
    print(f"Current = {I2C()[3]}, Target = {expected_bearing}, Change = {angle}")
    if direction == "L":
        ser.write(b"405,105\n")
        if I2C()[3] > expected_bearing:
            while I2C()[3] > expected_bearing: pass
        while I2C()[3] < expected_bearing: pass
        ser.write(b"255,255\n")
    else:
        ser.write(b"105,405\n")
        if I2C()[3] < expected_bearing:
            while I2C()[3] < expected_bearing: pass
        while I2C()[3] > expected_bearing: pass
        ser.write(b"255,255\n")
    if abs(I2C()[3] - expected_bearing) > 10:
        if direction == "L":
            ser.write(b"185,325\n")
            if I2C()[3] < expected_bearing:
                while I2C()[3] < expected_bearing: pass
            while I2C()[3] > (expected_bearing - 1): pass
            ser.write(b"255,255\n")
        else:
            ser.write(b"325,185\n")
            if I2C()[3] > expected_bearing:
                while I2C()[3] > expected_bearing: pass
            while I2C()[3] < (expected_bearing + 1): pass
            ser.write(b"255,255\n")
    time.sleep(0.25)
    return

def move(distance, t):
    cms = abs(distance) / t
    absan = CMS2AN(cms)
    print(absan)
    #absan = distance/t * 10
    if distance < 0:
        actual = int(255 - absan)
    else:
        actual = int(255 + absan)
    ser.write(f"{actual:03},{actual:03}\n".encode('utf-8'))
    time.sleep(t)
    ser.write(b"255,255\n")
    time.sleep(0.5)
    return

def analyse_image(image):
    # Convert to grayscale, slight brightness shift if needed
    #   Adjust alpha, beta to taste
    #   Higher alpha -> stronger contrast, changed to reduce glare washout
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    removed = int(kr * imagetemp.shape[0])
    img = imagetemp[int(kr * imagetemp.shape[0]):]
    rows, cols, _ = img.shape

    # Additional morphological steps to remove small bright specks
    # that can cause glare confusion
    kernel = np.ones((3, 3), np.uint8)

    ######################
    ### GENERATE BLACK ###
    ######################
    contourimg = img.copy()

    # Convert to grayscale and do morphological open to remove noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bcx, bcy = 0, 0
    ret, btemplate = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Morph open to remove noise, morph close to close up the line
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_OPEN, kernel)
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_CLOSE, kernel)
    ret, btemplate = cv2.threshold(btemplate, 128, 255, cv2.THRESH_BINARY_INV)
    # Force edges to be white so they don't get picked
    for i in range(rows):
        for j in range(cols):
            if i <= 5 or i >= (rows - 5):
                btemplate[i][j] = 255
            if j <= 5 or j >= (cols - 5):
                btemplate[i][j] = 255
    contours, hierarchy = cv2.findContours(btemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bcxlist = bcylist = bcc = 0
    centers, weights = [], []
    for index in range(len(contours)):
        cnt = contours[index]
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            if y + h > 0.75 * rows:
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.drawContours(contourimg, [cnt], -1, (79, 127, 247, 255), 2)
                cv2.circle(contourimg, (cX, cY), 7, (79, 127, 247, 255), -1)
                centers.append([cY, cX])
                weights.append(cY / rows)
    try:
        weights.sort()
        centers.sort()
        k_weight = 1 / sum(weights)
        modweights = [weights[i] * k_weight for i in range(len(weights))]
        for i in range(len(centers)):
            bcxlist += centers[i][1] * modweights[i]
            bcylist += centers[i][0] * modweights[i]
            bcc += 1
        bcx, bcy = int(bcxlist // bcc), int(bcylist // bcc)
    except:
        bcx, bcy = int(cols // 2), int(rows // 2)
    try:
        if len(centers) > 1:
            cv2.circle(contourimg, (bcx, bcy), (0, 0, 255, 255), -1)
    except:
        pass
    to_return = ['black', bcx, bcy]
   
    #########################
    ### GET GREEN SQUARES ###
    #########################
    ghsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gtemplate = cv2.inRange(ghsvimg, GREEN_LOWER_THRESHOLD, GREEN_UPPER_THRESHOLD)
    gtemplate = cv2.morphologyEx(gtemplate, cv2.MORPH_OPEN, kernel)
    gtemplate = cv2.morphologyEx(gtemplate, cv2.MORPH_CLOSE, kernel)
    ret, gtemplate = cv2.threshold(gtemplate, 128, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(gtemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_squares = 0
    gcs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 850 < area < 3750:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(contourimg, [contour], -1, (255, 0, 0, 255), 2)
            cv2.circle(contourimg, (cX, cY), 7, (255, 0, 0, 255), -1)
            gcs.append([cX, cY])
            green_squares += 1

    # Green squares are present, do turning logic
    if green_squares:
        green_squares_present = [False, False, False, False]
        green_squares_coords = [[0, 0], [0, 0], [0, 0], [0, 0]]

        for gc in gcs:
            gx, gy = gc[0], gc[1]
            bgx = bgxc = bgy = bgyc = 0
            for x in range(len(btemplate[gy])):
                if btemplate[gy][x] < 128:
                    bgx += x
                    bgxc += 1
            # Examine black line vertically near centroid
            for y in range(len(btemplate)):
                if btemplate[y][gx] < 128:
                    bgy += y
                    bgyc += 1
            try:
                bgx = bgx // bgxc
            except:
                bgx = gx
            try:
                bgy = bgy // bgyc
            except:
                bgy = gy
            cv2.circle(contourimg, (gx, bgy), 7, (255, 0, 0, 255), -1)
            cv2.circle(contourimg, (bgx, gy), 7, (255, 0, 0, 255), -1)
            cv2.rectangle(contourimg, (min(gx, bgx), gy), (max(gx, bgx), gy), (255, 0, 0, 255), 5)
            cv2.rectangle(contourimg, (gx, min(gy, bgy)), (gx, max(gy, bgy)), (255, 0, 0, 255), 5)

            if gy > bgy:
                if gx > bgx:
                    green_squares_present[3] = True  # bottom right
                    green_squares_coords[3] = [gx, gy]
                else:
                    green_squares_present[2] = True  # bottom left
                    green_squares_coords[2] = [gx, gy]
            else:
                if gx > bgx:
                    green_squares_present[0] = True  # top right
                    green_squares_coords[0] = [gx, gy]
                else:
                    green_squares_present[1] = True  # top left
                    green_squares_coords[1] = [gx, gy]

        # Replace these parts with forward and turn
        if green_squares_present[3] and green_squares_present[2]:
            to_return = ['green', 'u', cols // 2, (green_squares_coords[3][1] + green_squares_coords[2][1]) // 2]
        elif green_squares_present[3]:
            to_return = ['green', 'r', 0, green_squares_coords[3][1]]
        elif green_squares_present[2]:
            to_return = ['green', 'l', 0, green_squares_coords[2][1]]
        else:
            to_return = ['black', cols // 2, rows // 2]
    

   
    ####################
    ### GET RED LINE ###
    ####################
    rhsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(rhsvimg, RED_LOWER_THRESHOLD_1, RED_UPPER_THRESHOLD_1)
    mask2 = cv2.inRange(rhsvimg, RED_LOWER_THRESHOLD_2, RED_UPPER_THRESHOLD_2)
    rtemplate = cv2.bitwise_or(mask1, mask2)
    rtemplate = cv2.morphologyEx(rtemplate, cv2.MORPH_OPEN, kernel)
    rtemplate = cv2.morphologyEx(rtemplate, cv2.MORPH_CLOSE, kernel)
    ret, rtemplate = cv2.threshold(rtemplate, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(rtemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 900:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(contourimg, [contour], -1, (0, 255, 0, 255), 2)
            cv2.circle(contourimg, (cX, cY), 7, (0, 255, 0, 255), -1)
            to_return = ['red', -1]
    cv2_imshow("Black", btemplate, 10, 70)
    cv2_imshow("Green", gtemplate, 410, 70)
    cv2_imshow("Red", rtemplate, 810, 70)
    cv2_imshow("Analysis", cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB), 410, 350)
    return to_return, rows, cols

######################
### MAIN GAME LOOP ###
######################
while True:
    if button.is_pressed:
        print("interrupt")
        ser.write(b"255,255\n")
        exit()
    f_dist = I2C()[0]
    if f_dist < OBSTACLE_DETECTION_THRESHOLD and started and f_dist > 0:
        print("Obstacle detected!")
        print(f_dist)
        ser.write(b"255,255\n")
        move(-8, 1.5)
        turn('r', 90)
        while True:
            move_until_detect('l')
            move_until_undetect('l')
            turn('l', 90)
            if check_for_line:
                move_until_detect('l')
                object_d = move_until_undetect('l')
                move_speed(205, object_d / 2)
                move(-1 * (ROBOT_LENGTH / 2), 2)
                turn('r', 90)
                move(5, 1)
                break
    st = time.time()
    dt = st - prev_time
    data, rows, cols = analyse_image(picam2.capture_array())
    print(f"FPS: {1 / (time.time() - st)}")
    if data[0] == 'red':
        print("Red found.")
        if started:
            started = False
            break
        else:
            ser.write(b"305,305\n")
    elif data[0] == 'green' and started:
        ser.write(b"255,255\n")
        centroid = data[2:]
        actual_y_distance = 19.38859**(1 - (centroid[1] + kr * 194) / (rows + kr * 194)) + 3.92148
        move(actual_y_distance * 1.3, actual_y_distance * 1.3 / 5)
        if data[1] == "u":
            turn('r', 180)
        else:
            turn(data[1], 90)
        move(5, 1)
    elif data[0] == 'black' and started:
        centroid = data[1:]
        print(centroid)
        actual_y_distance = 19.38859**(1 - (centroid[1] + kr * 194) / (rows + kr * 194)) + 3.92148
        max_x_error = actual_y_distance * 0.366399 + 2.25391
        actual_x_distance = abs(centroid[0] - cols / 2) / (cols / 2) * max_x_error
        actual_x_distance *= -1 if centroid[0] < cols / 2 else 1
        print(actual_x_distance, actual_y_distance)
        error_x = -1 * actual_x_distance
        d_error = (error_x - prev_error) / dt if dt > 0 else 0.0
        u = (kp * error_x) + (ki * error_sum) + (kd * d_error)
        prev_error = error_x
        prev_time = st
        base_speed = BASE
        motor_right_speed = base_speed - u
        motor_left_speed = base_speed + u
        motor_left_speed = min(255, max(motor_left_speed, -255))
        motor_right_speed = min(255, max(motor_right_speed, -255))
        motor_left_speed += 255
        motor_right_speed += 255
        print(f"Damn = {int(motor_left_speed)},{int(motor_right_speed)}\n")
        ser.write(f"{int(motor_left_speed)},{int(motor_right_speed)}\n".encode("utf-8"))
    time.sleep(0.1)

# Clean up: move forward for a set distance
move(10, 2)
F_SENSOR.stop_ranging()
R_SENSOR.stop_ranging()
L_SENSOR.stop_ranging()

F_SENSOR.close()
R_SENSOR.close()
L_SENSOR.close()
exit()
