import cv2, os
from matplotlib import pyplot as plt
import numpy as np
path = os.getcwd()
wrong = []
num_correct = 0
"""
Testcases: 
001 to 008: Simple line following
009 to 023: 4-way intersections
024 to 032: 'T' intersections
032 to 043: Turns involving curves
044 to 058: Cases where the next intersection is spotted
059 to 060: Cases where the future line is spotted
"""
data = {
    '001.png': 'Decision: Follow blue dot',
    '002.png': 'Decision: Follow blue dot',
    '003.png': 'Decision: Follow blue dot',
    '004.png': 'Decision: Follow blue dot',
    '005.png': 'Decision: Follow blue dot',
    '006.png': 'Decision: Follow blue dot',
    '007.png': 'Decision: Follow blue dot',
    '008.png': 'Decision: Follow blue dot',
    '009.png': 'Decision: Turn Right',
    '010.png': 'Decision: Turn Left',
    '011.png': 'Decision: Continue Forward',
    '012.png': 'Decision: Continue Forward',
    '013.png': 'Decision: U-Turn',
    '014.png': 'Decision: Turn Right',
    '015.png': 'Decision: Turn Left',
    '016.png': 'Decision: Continue Forward',
    '017.png': 'Decision: Turn Right',
    '018.png': 'Decision: Turn Left',
    '019.png': 'Decision: Turn Right',
    '020.png': 'Decision: Turn Left',
    '021.png': 'Decision: U-Turn',
    '022.png': 'Decision: U-Turn',
    '023.png': 'Decision: U-Turn',
    '024.png': 'Decision: Turn Right',
    '025.png': 'Decision: Turn Left',
    '026.png': 'Decision: U-Turn',
    '027.png': 'Decision: Turn Right',
    '028.png': 'Decision: Continue Forward',
    '029.png': 'Decision: Turn Right',
    '030.png': 'Decision: Turn Left',
    '031.png': 'Decision: Continue Forward',
    '032.png': 'Decision: Turn Left',
    '033.png': 'Decision: Turn Right',
    '034.png': 'Decision: Turn Left',
    '035.png': 'Decision: U-Turn',
    '036.png': 'Decision: Follow blue dot',
    '037.png': 'Decision: Turn Right',
    '038.png': 'Decision: Turn Right',
    '039.png': 'Decision: Continue Forward',
    '040.png': 'Decision: Follow blue dot',
    '041.png': 'Decision: Turn Left',
    '042.png': 'Decision: Turn Left',
    '043.png': 'Decision: Continue Forward',
    '044.png': 'Decision: Turn Left',
    '045.png': 'Decision: Turn Right',
    '046.png': 'Decision: Continue Forward',
    '047.png': 'Decision: Continue Forward',
    '048.png': 'Decision: U-Turn',
    '049.png': 'Decision: Turn Right',
    '050.png': 'Decision: Turn Left',
    '051.png': 'Decision: Turn Left',
    '052.png': 'Decision: Turn Right',
    '053.png': 'Decision: Continue Forward',
    '054.png': 'Decision: Turn Right',
    '055.png': 'Decision: Turn Left',
    '056.png': 'Decision: U-Turn',
    '057.png': 'Decision: U-Turn',
    '058.png': 'Decision: U-Turn',
    '059.png': 'Decision: Follow blue dot',
    '060.png': 'Decision: Follow blue dot'
}
for (root,dirs,files) in os.walk(path + '/testcases'):
    filetemp = files[:]
    filetemp.sort()
    for file in filetemp:
        decision = ""
        if file == ".DS_Store": continue
        img1 = cv2.imread(path + "/testcases/" + file)
        img = cv2.convertScaleAbs(img1, alpha=3.0, beta=0.9)
        rows,cols,_ = img.shape
        
        
        ######################
        ### GENERATE BLACK ###
        ######################
        mod = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret,btemplate = cv2.threshold(gray, 100, 255, 0)
        contourimg = img.copy()
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
        for y_change in range(-10, 11):
            for x_change in range(-10, 11):     
                mod[(bcy + y_change, bcx + x_change)] = [247, 127, 79]
        ######################
        ### GENERATE GREEN ###
        ######################
        temp = img.copy()
        black_pixels = np.where(
            (img[:, :, 0] >= 150) & 
            (img[:, :, 1] >= 150) & 
            (img[:, :, 2] >= 150) 
        )
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
                #cv2.rectangle(mod,(x + w//2 - 5, y + h//2 - 5),(x + w//2 + 5,y + h//2 + 5), (0, 0, 255), -1)
                green_squares += 1

        #####################
        ### TURNING LOGIC ###
        #####################
        if green_squares:
            for gc in gcs:
                gcx,gcy = gc[0],gc[1]
                for x_change in range(-10, 11):
                    for y_change in range(-10, 11):
                        mod[(gcy + y_change, gcx + x_change)] = [0, 0, 255]
            green_squares_present = [False] * 4
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
                #print(gx, gy)
                #print(bgx, gy)
                #print(gx, bgy)
                for x_change in range(-10, 11):
                    for y_change in range(-10, 11):
                        mod[(gy + y_change, bgx + x_change)] = [0, 0, 255]
                        mod[(bgy + y_change, gx + x_change)] = [0, 0, 255]
                cv2.rectangle(mod,(min(gx,bgx),gy),(max(gx,bgx),gy),(0,0,255),10)
                cv2.rectangle(mod,(gx,min(gy,bgy)),(gx,max(gy,bgy)),(0,0,255),10)
                if gy > bgy:
                    if gx > bgx: 
                        green_squares_present[3] = True
                    else: 
                        green_squares_present[2] = True
                else:
                    if gx > bgx: 
                        green_squares_present[0] = True
                    else: 
                        green_squares_present[1] = True
            if green_squares_present[3] and green_squares_present[2]:
                decision = "Decision: U-Turn"
            elif green_squares_present[3]:
                decision = "Decision: Turn Right"
            elif green_squares_present[2]:
                decision = "Decision: Turn Left"
            else:
                decision = "Decision: Continue Forward" 
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Contrast Image')
            plt.xlabel(f"{decision}\nExpected {data[file]}")

            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
            plt.title('Binary Image')

            plt.subplot(2, 3, 4)
            plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            plt.title('Green Binary Image')

            plt.subplot(2, 3, 5)
            plt.imshow(cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
            plt.title('Contours Image')

            plt.subplot(2, 3, 6)
            plt.imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB))
            plt.title('Analysed Image')
        else:
            decision = "Decision: Follow blue dot"
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            plt.subplot(1, 4, 2)
            plt.imshow(cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
            plt.xlabel(decision)
            plt.title('Binary Image')

            plt.subplot(1, 4, 3)
            plt.imshow(cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
            plt.xlabel(f'Expected {data[file]}')
            plt.title('Contours Image')

            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB))
            plt.title('Analysed Image')
        plt.savefig(path + "/images/" + file)
        plt.close()
        correct = decision==data[file]
        if correct: num_correct += 1
        else: wrong.append(file)
        print(f"File: {file}\n{decision}\nExpected {data[file]}\n{'Correct' if correct else 'Wrong'}")
print(f"{num_correct} of 60 testcases passed!\nIncorrect questions: ")
for wrong_qn in wrong:
    print(wrong_qn)