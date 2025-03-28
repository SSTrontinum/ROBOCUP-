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

# Set camera controls if needed (example commented out)
# picam2.set_controls({
#     "AwbEnable": False,
#     "AnalogueGain": 5.0,
#     "ExposureTime": 1000000
# })

picam2.start(show_preview=True)

# PID config
kp = 1
ki = 0
kd = 0
error_sum = 0
prev_error = 0
kr = 0.25  # original cropping factor for top offset
prev_time = time.time()

def analyse_image(image):
    # Adjust image contrast to reduce glare effects
    imagetemp = cv2.convertScaleAbs(image, alpha=1.5, beta=0.1)
    # --- Adjust ROI to bottom section for immediate control ---
    # For example, select the bottom 50% of the image:
    height = imagetemp.shape[0]
    roi = imagetemp[int(height*0.5):, :]
    
    # Convert ROI to grayscale and apply morphological opening to remove noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # --- Thresholding to obtain binary image of the line ---
    threshold_value = 140  # adjust as needed
    ret, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # --- Slice-Based Centroid Calculation ---
    num_slices = 5  # divide ROI into 5 horizontal slices
    slice_height = binary.shape[0] // num_slices
    slice_centroids = []
    weights = []
    
    for i in range(num_slices):
        # Define slice: from bottom slice (i=0) upward (i=num_slices-1)
        y_start = binary.shape[0] - (i+1)*slice_height
        y_end = binary.shape[0] - i*slice_height
        slice_img = binary[y_start:y_end, :]
        
        # Calculate moments for each slice
        M = cv2.moments(slice_img)
        if M['m00'] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + y_start  # offset with ROI position
            slice_centroids.append((cx, cy))
            # Weight: slices nearer bottom (lower y) get higher weight
            weight = (i+1)  # or use an inverse function if desired
            weights.append(weight)
        else:
            # In case no line is detected in this slice, use center of slice
            cx = binary.shape[1] // 2
            cy = (y_start + y_end) // 2
            slice_centroids.append((cx, cy))
            weights.append(1)
    
    # Compute weighted average centroid from slices
    weighted_sum_x = sum(cx * w for (cx, _), w in zip(slice_centroids, weights))
    total_weight = sum(weights)
    global_cx = int(weighted_sum_x / total_weight)
    
    # --- Polynomial Curve Fitting ---
    # Use detected centroids as points for fitting a second order polynomial.
    if len(slice_centroids) >= 3:
        # y-coordinates become independent variable
        pts = np.array(slice_centroids)
        xs = pts[:,0]
        ys = pts[:,1]
        # Fit polynomial: x = a*y^2 + b*y + c
        poly_coeff = np.polyfit(ys, xs, 2)
        poly = np.poly1d(poly_coeff)
        # Choose a target point at a future y-position in ROI (e.g. near bottom)
        target_y = binary.shape[0] - 1 + int(height*0.5)  # adjust as necessary
        target_x = int(poly(target_y))
    else:
        # Fallback if not enough points
        target_x = global_cx
        target_y = roi.shape[0] - 1 + int(height*0.5)
    
    # --- Display for debugging ---
    contour_img = roi.copy()
    for (cx, cy) in slice_centroids:
        cv2.circle(contour_img, (cx, cy - int(height*0.5)), 3, (0, 255, 0), -1)
    # Mark global weighted centroid in blue
    cv2.circle(contour_img, (global_cx, binary.shape[0] - 1), 5, (255, 0, 0), -1)
    # Mark target point from polynomial fit in red
    cv2.circle(contour_img, (target_x, target_y - int(height*0.5)), 5, (0, 0, 255), -1)
    cv2.imshow("Analysis", contour_img)
    
    # Return target point (from polynomial fit) and other data as needed
    return (target_x, target_y), roi.shape[0], binary.shape[1]

while True:
    st = time.time()
    dt = st - prev_time
    # Capture image array from Picamera2
    frame = picam2.capture_array()
    centroid, rows, cols = analyse_image(frame)

    # Debug print frame rate
    print(f"FPS: {1 / (time.time() - st)}")

    # Use the x-coordinate of the target point for error calculation
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)

cv2.destroyAllWindows()
