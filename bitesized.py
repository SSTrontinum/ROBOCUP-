import cv2, os
import numpy as np
path = os.getcwd()
filepath = 'testcases/046.png'
img1 = cv2.imread(filepath)

def analyse_image(image):
    img = cv2.convertScaleAbs(image, alpha=4.0, beta=0.8)
    rows,cols,_ = img.shape
    contourimg = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret,btemplate = cv2.threshold(gray, 100, 255, 0) 
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
            cv2.rectangle(contourimg,(x,y),(x+w,y+h),(255,0,255),5)
            bcxlist += x + w//2
            bcylist += y + h//2
            bcc += 1
    bcx, bcy = bcxlist // bcc, bcylist // bcc
    temp = img.copy()
    black_pixels = np.where((img[:, :, 0] >= 150) & (img[:, :, 1] >= 150) & (img[:, :, 2] >= 150) )
    temp[black_pixels] = [0, 0, 0]
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret,template = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(template,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    green_squares = 0
    gcs = []
    for contour in contours[1:]:
        if cv2.contourArea(contour) > 5000:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(contourimg, (x, y), (x + w, y + h),(0, 0, 255),2)
            gcs.append([x + w//2, y + h//2])
            green_squares += 1
    if green_squares:
        green_squares_present = [False] * 4
        green_squares_coors = [[0, 0]] * 4
        for gc in gcs:
            gx, gy = gc[0], gc[1]
            bgx = bgxc = bgy = bgyc = 0
            for x in range(len(btemplate[gy])):
                if btemplate[gy][x] < 100 and abs(x - gx) <= 200:
                    bgx += x
                    bgxc += 1
            for y in range(len(btemplate)):
                if btemplate[y][gx] < 100 and abs(y - gy) <= 200:
                    bgy += y
                    bgyc += 1
            bgx = bgx // bgxc
            bgy = bgy // bgyc
            if gy > bgy:
                if gx > bgx: 
                    green_squares_present[3] = True
                    green_squares_coords[3] = [gx,gy]
                else: 
                    green_squares_present[2] = True
                    green_squares_coords[2] = [gx,gy]
            else:
                if gx > bgx:
                    green_squares_present[0] = True
                    green_squares_coords[0] = [gx,gy]
                else: =
                    green_squares_present[1] = True
                    green_squares_coords[1] = [gx,gy]
        if green_squares_present[3] and green_squares_present[2]:
            return [-1]
        elif green_squares_present[3]:
            return green_squares_coords[2]
        elif green_squares_present[2]:
            return green_squares_coords[2]
        else:
            return [bcx,bcy]
    else:
        return [bcx,bcy]