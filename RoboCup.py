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
picam2.set_controls({
    "AwbEnable": False,
    "AnalogueGain": 5.0,      # Lower gain
    "ExposureTime": 1000000     # Example exposure in microseconds
})

picam2.start(show_preview=True)

plot = True
disp = True
home = os.environ["HOME"]

# PID config
kp = 1.5
ki = 0
kd = 0
error_sum = 0
prev_error = 0
prev_time = time.time()

def analyse_image(image):
    # Convert to grayscale, slight brightness shift if needed
    #   Adjust alpha, beta to taste
    #   Higher alpha -> stronger contrast, changed to reduce glare washout
    #img = cv2.convertScaleAbs(image, alpha=1.0, beta=0.1)
    img= image
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
    threshold = 100
    while True:
        try:
            ret,btemplate = cv2.threshold(gray, threshold, 255, 0)
            # Force edges to be white so they don't get picked
            for i in range(rows):
                for j in range(cols):
                    if i <= 10 or i >= (rows - 10):
                        btemplate[i][j] = 255
                    if j <= 10 or j >= (cols - 10):
                        btemplate[i][j] = 255

            # Another morphological close to merge broken black-line parts
            btemplate = cv2.morphologyEx(btemplate, cv2.MORPH_CLOSE, kernel)

            contours, hierarchy = cv2.findContours(btemplate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bcxlist = bcylist = 0
            bcc = 0
            for index in range(len(contours)):
                cnt = contours[index]
                if cv2.contourArea(cnt) > 100:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if plot:
                        cv2.rectangle(contourimg, (x, y), (x + w, y + h),
                                      (247, 127, 79, 255), 2)
                    bcxlist += x + w//2
                    bcylist += y + h//2
                    bcc += 1
            bcx, bcy = bcxlist // bcc, bcylist // bcc
        except:
            threshold += 10
        else:
            break

    # Draw the black-line centroid as a blue point in mod
    if plot or disp:
        for y_change in range(-5, 6):
            for x_change in range(-5, 6):
                try:
                    mod[(bcy + y_change, bcx + x_change)] = [79, 127, 247, 255]
                except:
                    pass

    ######################
    ### GENERATE GREEN ###
    ######################
    # by blackening out high-luminance and low-luminance ranges
    temp = img.copy()
    # White or near-white pixels set to black
    white_pixels = np.where(
        (img[:, :, 0] >= 170) &
        (img[:, :, 1] >= 170) &
        (img[:, :, 2] >= 170)
    )
    temp[white_pixels] = [0, 0, 0, 255]
    # Dark or near-black pixels set to black
    black_pixels = np.where(
        (img[:, :, 0] <= 125) &
        (img[:, :, 1] <= 125) &
        (img[:, :, 2] <= 125)
    )
    temp[black_pixels] = [0, 0, 0, 255]

    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret, template = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

    # Additional morph-op to consolidate green squares
    template = cv2.morphologyEx(template, cv2.MORPH_OPEN, kernel)
    template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_squares = 0
    gcs = []  # Green centroids

    # Skip first contour to ignore entire frame contour
    for contour in contours[1:]:
        area = cv2.contourArea(contour)
        if 100 < area < 500:
            x, y, w, h = cv2.boundingRect(contour)
            if plot:
                cv2.rectangle(contourimg, (x, y), (x + w, y + h),
                              (0, 0, 255, 255), 2)
            gcs.append([x + w//2, y + h//2])
            green_squares += 1

    #####################
    ### TURNING LOGIC ###
    #####################
    if green_squares:
        if plot or disp:
            for gc in gcs:
                gcx, gcy = gc
                for x_change in range(-5, 6):
                    for y_change in range(-5, 6):
                        try:
                            mod[(gcy + y_change, gcx + x_change)] = [0, 0, 255, 255]
                        except:
                            pass

        green_squares_present = [False, False, False, False]
        green_squares_coords = [[0, 0], [0, 0], [0, 0], [0, 0]]

        for gc in gcs:
            gx, gy = gc[0], gc[1]
            bgx = bgxc = bgy = bgyc = 0

            # Examine black line horizontally near centroid
            for x in range(len(btemplate[gy])):
                if btemplate[gy][x] < threshold and abs(x - gx) <= 150:
                    bgx += x
                    bgxc += 1
            # Examine black line vertically near centroid
            for yval in range(len(btemplate)):
                if btemplate[yval][gx] < threshold and abs(yval - gy) <= 150:
                    bgy += yval
                    bgyc += 1
            try:
                bgx = bgx // bgxc
            except:
                bgx = gx
            try:
                bgy = bgy // bgyc
            except:
                bgy = gy

            if plot or disp:
                for x_change in range(-5, 6):
                    for y_change in range(-5, 6):
                        try:
                            mod[(gy + y_change, bgx + x_change)] = [255, 0, 0, 255]
                            mod[(bgy + y_change, gx + x_change)] = [255, 0, 0, 255]
                        except:
                            pass
                cv2.rectangle(mod, (min(gx, bgx), gy), (max(gx, bgx), gy), (255, 0, 0, 255), 2)
                cv2.rectangle(mod, (gx, min(gy, bgy)), (gx, max(gy, bgy)), (255, 0, 0, 255), 2)

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
            to_return[0] += 50
        elif green_squares_present[2]:
            decision = "Decision: Turn Left"
            to_return = green_squares_coords[2]
            to_return[0] -= 50
        else:
            decision = "Decision: Continue Forward"
            to_return = [bcx, bcy]

        if plot:
            # Plot resulting images for debugging
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Contrast & Morph Fix')
            plt.xlabel(decision)

            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
            plt.title('Binary Image')

            plt.subplot(2, 3, 4)
            plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            plt.title('Green Binary')

            plt.subplot(2, 3, 5)
            plt.imshow(cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
            plt.title('Contours')

            plt.subplot(2, 3, 6)
            plt.imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB))
            plt.title('Analysed Result')
            plt.show()

        cv2.imshow("Analysis", cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
        return to_return, rows, cols

    else:
        # No green squares found: follow the blue dot
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

        cv2.imshow("Analysis", cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
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
        """
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
        """

    if plot:
        break
    time.sleep(0.1)