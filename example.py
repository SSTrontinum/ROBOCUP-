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
#    "AwbEnable": False,
#    "AnalogueGain": 5.0,      # Lower gain
#    "ExposureTime": 1000000   # Example exposure in microseconds
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
    # Adjust contrast to reduce glare
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    cv2.imshow('test', imagetemp)
    img = imagetemp[int(kr * imagetemp.shape[0]):]
    rows, cols, _ = img.shape

    kernel = np.ones((3, 3), np.uint8)

    if plot or disp:
        mod = img.copy()       # Final analysed image for showing
        contourimg = img.copy() # Image showing all contours

    # Convert to grayscale and remove noise with morphological open
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Thresholding to obtain binary image of the line
    threshold = 140
    ret, btemplate = cv2.threshold(gray, threshold, 255, 0)
    # Force edges to be white so they aren't picked up as part of the line
    for i in range(rows):
        for j in range(cols):
            if i <= 5 or i >= (rows - 5):
                btemplate[i][j] = 255
            if j <= 5 or j >= (cols - 5):
                btemplate[i][j] = 255

    # Merge broken black-line parts with morphological close
    btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(btemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bcxlist = bcylist = 0
    bcc = 0
    centers = []
    weights = []
    # Collect centres of contours that fall within area limits
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 30000:
            if plot or disp:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.drawContours(contourimg, [cnt], -1, (79, 127, 247, 255), 2)
                    cv2.circle(contourimg, (cX, cY), 7, (79, 127, 247, 255), -1)
            # Append centre as [y, x] for polynomial fit (y is independent variable)
            centers.append([int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])])
            weights.append((int(M["m01"] / M["m00"])) / rows)

    try:
        # Compute weighted average for a global centroid (fallback)
        weights.sort()
        centers.sort()
        k_weight = 1 / sum(weights)
        modweights = [w * k_weight for w in weights]
        for i in range(len(centers)):
            bcxlist += centers[i][1] * modweights[i]
            bcylist += centers[i][0] * modweights[i]
            bcc += 1
        bcx, bcy = bcxlist // bcc, bcylist // bcc
    except:
        bcx, bcy = cols // 2, rows // 2

    # Draw small blue square around the global centroid
    for y_change in range(-4, 5):
        for x_change in range(-4, 5):
            contourimg[min(max(bcy + y_change, 0), rows-1)][min(max(bcx + x_change, 0), cols-1)] = [255, 0, 0, 255]

    # POLYNOMAIL
    # Use centres if at least three points are available
    if len(centers) >= 3:
        pts = np.array(centers)
        # Sort points by y-coordinate 
        pts = pts[pts[:, 0].argsort()]
        ys = pts[:, 0]
        xs = pts[:, 1]
        # x = a*y^2 + b*y + c
        poly_coeff = np.polyfit(ys, xs, 2)
        poly = np.poly1d(poly_coeff)
        # Select target at the bottom of the image
        target_y = rows - 1
        target_x = int(poly(target_y))
        # Draw target point in red on contour image
        cv2.circle(contourimg, (target_x, target_y), 5, (0, 0, 255), -1)
    else:
        target_x, target_y = bcx, bcy

    if plot:
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
    
    # Return the target point from polynomial fit and image dimensions for PID control
    return [target_x, target_y], rows, cols

plot = False
while True:
    st = time.time()
    dt = st - prev_time
    centroid, rows, cols = analyse_image(picam2.capture_array())

    print(f"FPS: {1 / (time.time() - st)}")

    if centroid[0] == -1:
        print("U-turn")
        # Implement U-turn logic here
    else:
        print(centroid)
        # Use x-coordinate of target point for error calculation
        error_x = centroid[0] - (cols / 2)
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
