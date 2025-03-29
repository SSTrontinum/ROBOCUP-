import cv2, os, serial, time
from matplotlib import pyplot as plt
import numpy as np
from picamzero import Camera
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
ser.reset_input_buffer()
ser.flush()
home = os.environ['HOME']
cam = Camera()
IMAGE = "frame.jpg"
path = os.getcwd()
filepath = f"{path}/{IMAGE}"

# Take frame
def new_frame():
    cam.take_photo(filepath)

def analyse_image(image):
    img = cv2.convertScaleAbs(image, alpha=4.0, beta=0.8)
    rows,cols,_ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    threshold = 100
    while True:
        try:
            ret,btemplate = cv2.threshold(gray, threshold, 255, 0) 
            for i in range(rows): 
                for j in range(cols):
                    if i <= 10 or i >= (rows - 10):
                        btemplate[i][j] = 255
                    if j <= 10 or j >= (cols - 10):
                        btemplate[i][j] = 255
            contours,hierarchy = cv2.findContours(btemplate,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bcxlist = bcylist = bcc = 0
            for index in range(len(contours)):
                cnt=contours[index]
                if cv2.contourArea(cnt) > 5000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    bcxlist += x + w//2
                    bcylist += y + h//2
                    bcc += 1
            bcx, bcy = bcxlist // bcc, bcylist // bcc
        except:
            threshold += 10
        else: break
    temp = img.copy()
    black_pixels = np.where((img[:, :, 0] >= 150) & (img[:, :, 1] >= 150) & (img[:, :, 2] >= 150) )
    temp[black_pixels] = [0, 0, 0]
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret,template = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(template,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_squares, gcs = 0, []
    for contour in contours[1:]:
        if cv2.contourArea(contour) > 5000:
            x,y,w,h = cv2.boundingRect(contour)
            gcs.append([x + w//2, y + h//2])
            green_squares += 1
    if green_squares:
        green_squares_present, green_squares_coords = [False, False, False, False], [[0, 0], [0, 0], [0, 0], [0, 0]]
        for gc in gcs:
            gx, gy = gc[0], gc[1]
            bgx = bgxc = bgy = bgyc = 0
            for x in range(len(btemplate[gy])):
                if btemplate[gy][x] < 100 and abs(x - gx) <= 200: bgx += x; bgxc += 1
            for y in range(len(btemplate)):
                if btemplate[y][gx] < 100 and abs(y - gy) <= 200: bgy += y; bgyc += 1
            bgx, bgy = bgx // bgxc, bgy // bgyc
            if gy > bgy:
                if gx > bgx: green_squares_present[3] = True; green_squares_coords[3] = [gx,gy]
                else: green_squares_present[2] = True; green_squares_coords[2] = [gx,gy]
            else:
                if gx > bgx: green_squares_present[0] = True; green_squares_coords[0] = [gx,gy]
                else: green_squares_present[1] = True; green_squares_coords[1] = [gx,gy]
        if green_squares_present[3] and green_squares_present[2]:
            return [-1, -1]
        elif green_squares_present[3]:
            return green_squares_coords[2]
        elif green_squares_present[2]:
            return green_squares_coords[2]
        else:
            return [bcx,bcy]
    else:
        return [bcx,bcy]

# Camera parameters
while True:
    new_frame()
    centroid = analyse_image(cv2.imread(filepath))
    if centroid[0] == -1:
        # U_turn
    else:
        # Complicated math
        print(centroid)
    time.sleep(0.1)