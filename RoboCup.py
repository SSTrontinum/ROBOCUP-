import cv2, os, serial, time
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

plot = True
disp = True
home = os.environ["HOME"]

# PID config
kp = 1
ki = 0
kd = 0
error_sum = 0
prev_error = 0
kr = 0.25
prev_time = time.time()

def analyse_image(image):
    # Convert to grayscale, slight brightness shift if needed
    #   Adjust alpha, beta to taste
    #   Higher alpha -> stronger contrast, changed to reduce glare washout
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    cv2.imshow('test', imagetemp)
    img = imagetemp[int(kr*imagetemp.shape[0]):]
    rows, cols, _ = img.shape

    # Additional morphological steps to remove small bright specks
    # that can cause glare confusion
    kernel = np.ones((3, 3), np.uint8)

    ######################
    ### GENERATE BLACK ###
    ######################
    if plot or disp:
        mod = img.copy()       # final analysed image for showing
        contourimg = img.copy() # image showing all contours

    # Convert to grayscale and do morphological open to remove noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Attempt a threshold that deals with partial glare
    # Increase threshold if result doesn't find line
    threshold = 140
    bcx, bcy = 0,0
    while True:
        try:
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
            for index in range(len(contours)):
                cnt = contours[index]
                if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 30000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if plot or disp:
                        M = cv2.moments(cnt)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # draw the contour and center of the shape on the image
                        cv2.drawContours(contourimg, [cnt], -1, (79, 127, 247, 255), 2)
                        cv2.circle(contourimg, (cX, cY), 7, (79, 127, 247, 255), -1)
                        #cv2.rectangle(contourimg, (x, y), (x + w, y + h),(79, 127, 247, 255), 2)
                    bcxlist += cX * (cX/rows)
                    bcylist += cY * (cY/cols)
                    bcc += 1
            bcx, bcy = bcxlist // bcc, bcylist // bcc
        except:
            threshold += 10
        else:
              break
    # Draw the black-line centroid as a blue point in mod
    if plot:
        decision = "Decision: Follow blue dot"
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 5, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Contrast & Morph Fix')

        plt.subplot(1, 5, 3)
        plt.imshow(cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
        plt.title('Binary Image')

        plt.subplot(1, 5, 4)
        plt.imshow(cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
        plt.title('Contours Image')

        plt.subplot(1, 5, 5)
        plt.imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB))
        plt.title('Analysed')
        plt.show()
    cv2.imshow("Analysis", cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
    return [bcx, bcy], rows, cols


plot = False
while True:
    st = time.time()
    dt = st - prev_time
    centroid, rows, cols = analyse_image(picam2.capture_array())

    print(f"FPS: {1 / (time.time() - st)}")

    if centroid[0] == -1:
        print("U-turn")
        # Implement U-turn logic here
        pass
    else:
        print(centroid)
        # Additional PID logic may be placed here.
        error_x = centroid[0] - (cols/2)
        d_error = (error_x - prev_error) / dt if dt > 0 else 0.0
        u = (kp * error_x) + (ki * error_sum) + (kd * d_error)
        prev_error = error_x
        prev_time = st

        base_speed = 150
        motor_right_speed = base_speed - u
        motor_left_speed = base_speed + u

        motor_left_speed = int(max(0, min(255, motor_left_speed))) + 255
        motor_right_speed = int(max(0, min(255, motor_right_speed))) + 255
        ser.write(f"{motor_left_speed},{motor_right_speed}\n".encode("utf-8"))

    if plot:
        break
    time.sleep(0.1)

