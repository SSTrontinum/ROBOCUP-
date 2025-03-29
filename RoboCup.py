import cv2, os, serial, time, VL53L0X
from matplotlib import pyplot as plt
from libcamera import Transform, controls
import numpy as np
from picamera2 import Picamera2, Preview

# Set up serial connection
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()

picam2 = Picamera2()

# Preview configuration (low resolution to reduce processing overhead)
preview_config = picam2.create_preview_configuration(
    {"size": (258, 194)},
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(preview_config)

# ---------------------------
# Key changes for glare fix:
#   Disable auto white balance
#   Set explicit exposure time
#   Set lower analogue gain
#   (Adjust as needed for environment)
# ---------------------------
#picam2.set_controls({
    #"AwbEnable": False,
    #"AnalogueGain": 5.0,      # Lower gain
    #"ExposureTime": 1000000     # Example exposure in microseconds
#})

picam2.start(show_preview=True)
home = os.environ["HOME"]

# Start I2C TOF devices
f_sensor = VL53L0X.VL53L0X(tca9548a_num=0, tca9548a_addr=0x70)
r_sensor = VL53L0X.VL53L0X(tca9548a_num=1, tca9548a_addr=0x70)
l_sensor = VL53L0X.VL53L0X(tca9548a_num=2, tca9548a_addr=0x70)
f_sensor.open()
r_sensor.open()
l_sensor.open()
f_sensor.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
r_sensor.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
l_sensor.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
ferror, rerror, lerror = 36, 52, 51

# PID config
kp = 1
ki = 0
kd = 0
error_sum = 0
prev_error = 0
kr = 0.25
prev_time = time.time()
frames = 0
green_lower_threshold = (36, 50, 25)
green_upper_threshold = (70, 255, 255)
red_lower_threshold = (0, 50, 25)
red_upper_threshold = (25, 255, 255)
started = False
# wheel diameter = 53.25 mm
# robot width = 114.00 mm

def turn(direction, angle):
    return 0

def move(distancet, time):
    cms_speed = distance / time
    analog_speed = 0.000296378 * cms_speed**5 - 0.0156624 * cms_speed**4 + 0.303438 * cms_speed**3 - 2.3577 * cms_speed**2 + 8.83765 * cms_speed + 19.42041
    print(analog_speed)

def analyse_image(image):
    # Convert to grayscale, slight brightness shift if needed
    #   Adjust alpha, beta to taste
    #   Higher alpha -> stronger contrast, changed to reduce glare washout
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    removed = int(kr*imagetemp.shape[0])
    img = imagetemp[int(kr*imagetemp.shape[0]):]
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
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Attempt a threshold that deals with partial glare
    # Increase threshold if result doesn't find line
    threshold = 140
    bcx, bcy = 0,0
    ret,btemplate = cv2.threshold(gray, threshold, 255, 0)
    # Force edges to be white so they don't get picked
    for i in range(rows):
        for j in range(cols):
            if i <= 5 or i >= (rows - 5):
                btemplate[i][j] = 255
            if j <= 5 or j >= (cols - 5):
                btemplate[i][j] = 255
    # Another morphological close to merge broken black-line parts
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(btemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bcxlist = bcylist = 0
    bcc = 0
    nc = len(contours)
    centers = []
    weights = []
    for index in range(len(contours)):
        cnt = contours[index]
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 10000:
            x,y,w,h = cv2.boundingRect(cnt)
            if y + h > 0.75 * rows:
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(contourimg, [cnt], -1, (79, 127, 247, 255), 2)
                cv2.circle(contourimg, (cX, cY), 7, (79, 127, 247, 255), -1)
                #cv2.rectangle(contourimg, (x, y), (x + w, y + h),(79, 127, 247, 255), 2)
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
    except: bcx, bcy = int(cols // 2), int(rows // 2)
    try:
        if len(centers) > 1: cv2.circle(contourimg, (bcx,bcy), (0, 0, 255, 255) -1)
    except: pass
    to_return = [bcx, bcy]
    
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    template = cv2.inRange(hsvimg, (36, 50, 25), (70, 255,255))
    ret,template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY_INV)
    template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_squares = 0
    gcs = [] 
    for contour in contours:
        area = cv2.contourArea(contour)
        if 900 < area < 3600:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(contourimg, [contour], -1, (255, 0, 0, 255), 2)
            cv2.circle(contourimg, (cX, cY), 7, (255, 0, 0, 255), -1)
            gcs.append([cX, cY])
            green_squares += 1
            
    if green_squares:
        green_squares_present = [False, False, False, False]
        green_squares_coords = [[0, 0], [0, 0], [0, 0], [0, 0]]

        for gc in gcs:
            gx, gy = gc[0], gc[1]
            bgx = bgxc = bgy = bgyc = 0

            for x in range(len(btemplate[gy])):
                if btemplate[gy][x] < threshold:
                    bgx += x
                    bgxc += 1
            # Examine black line vertically near centroid
            for y in range(len(btemplate)):
                if btemplate[y][gx] < threshold:
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
                    # bottom right
                    green_squares_present[3] = True
                    green_squares_coords[3] = [gx, gy]
                else:
                    # bottom left
                    green_squares_present[2] = True
                    green_squares_coords[2] = [gx, gy]
            else:
                if gx > bgx:
                    # top right
                    green_squares_present[0] = True
                    green_squares_coords[0] = [gx, gy]
                else:
                    # top left
                    green_squares_present[1] = True
                    green_squares_coords[1] = [gx, gy]

        if green_squares_present[3] and green_squares_present[2]:
            decision = "Decision: U-Turn"
            to_return = [-1, -1]
        elif green_squares_present[3]:
            decision = "Decision: Turn Right"
            to_return = green_squares_coords[3]
            to_return[0] = cols
        elif green_squares_present[2]:
            to_return = green_squares_coords[2]
            to_return[0] = 0
        else:
            to_return = [cols//2, rows//2]
    
    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rtemplate = cv2.inRange(hsvimg, red_lower_threshold, red_upper_threshold)
    ret,rtemplate = cv2.threshold(rtemplate, 128, 255, cv2.THRESH_BINARY)
    rtemplate = cv2.morphologyEx(rtemplate, cv2.MORPH_CLOSE, kernel)
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
    #green = cv2.bitwise_and(img,img, mask= mask)
    #gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    #ret,template = cv2.threshold(gsray, 10, 255, 0)
    cv2.imshow("hsv", template)
    cv2.imshow("Analysis", cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
    return to_return, rows, cols
while True:
    fdistance = int(f_sensor.get_distance()) - ferror
    if fdistance < 40 and not started:
        print("Obstacle detected!")
        ser.write(b"255,255\n")
        exit()
    else:
        st = time.time()
        dt = st - prev_time
        frames += 1
        centroid, rows, cols = analyse_image(picam2.capture_array())
        print(f"FPS: {1 / (time.time() - st)}")
        if centroid[0] == 'red':
            print("Red found.")
            if started:
                started = False
                break
            else:
                move(10, 50)
                started = True
        if not started: continue
        if centroid[0] == -1 :
            print("U-turn")
            # Implement U-turn logic here
            pass
        else:
            print(centroid)
            actual_y_distance = 19.38859**(1 - (centroid[1]+kr*194)/(rows + kr*194)) + 3.92148
            max_x_error = actual_y_distance * 0.366399 + 2.25391
            actual_x_distance = abs(centroid[0] - cols/2)/(cols/2) * max_x_error
            actual_x_distance *= -1 if centroid[0] < cols/2 else 1
            print(actual_x_distance, actual_y_distance)
            # Additional PID logic may be placed here.
            """
            error_x = -1 * actual_x_distance
            d_error = (error_x - prev_error) / dt if dt > 0 else 0.0
            u = (kp * error_x) + (ki * error_sum) + (kd * d_error)
            prev_error = error_x
            prev_time = st

            base_speed = 50
            motor_right_speed = base_speed - u
            motor_left_speed = base_speed + u

            motor_left_speed = int(max(0, min(255, motor_left_speed))) + 255
            motor_right_speed = int(max(0, min(255, motor_right_speed))) + 255
            ser.write(f"{motor_left_speed},{motor_right_speed}\n".encode("utf-8"))
            """
    time.sleep(0.1)

# Clean up: move forward for a set distance
move(10, 50)
f_dist.stop_ranging()
r_dist.stop_ranging()
l_dist.stop_ranging()

f_dist.close()
r_dist.close()
l_dist.close()