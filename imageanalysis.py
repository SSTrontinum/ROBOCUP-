import cv2, os
from matplotlib import pyplot as plt
import numpy as np
path = os.getcwd()
filepath = 'testcases/046.png'
"""
Testcases: 
001 to 008: Simple line following
009 to 023: 4-way intersections
024 to 032: 'T' intersections
032 to 043: Turns involving curves
044 to 058: Cases where the next intersection is spotted
059 to 060: Cases where the future line is spotted
"""
def analyse_image(image):
    img = cv2.convertScaleAbs(image, alpha=4.0, beta=0.8)
    rows,cols,_ = img.shape
    ######################
    ### GENERATE BLACK ###
    ######################
    mod = img.copy() # Mod is used to store the final analysed image
    contourimg = img.copy() # This will save the image that shows all the contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscales image
    ret,btemplate = cv2.threshold(gray, 100, 255, 0) # Forces a binary image so green becomes white
    for i in range(rows): # This generates a white border around the image so that the black lines will not reach the edges and mess with contouring
        for j in range(cols):
            if i <= 10 or i >= (rows - 10):
                btemplate[i][j] = 255
            if j <= 10 or j >= (cols - 10):
                btemplate[i][j] = 255
    contours,hierarchy = cv2.findContours(btemplate,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Finds contours of the binary image
    bcxlist = bcylist = bcc = 0 # The list of the centroids of the black contours, if more than one is found. Uses incremental mean to reduce time complexity
    # Finds the centroid of black contours
    for index in range(len(contours)):
        cnt=contours[index]
        if cv2.contourArea(cnt) > 5000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(contourimg,(x,y),(x+w,y+h),(255,0,255),5)
            bcxlist += x + w//2
            bcylist += y + h//2
            bcc += 1
    bcx, bcy = bcxlist // bcc, bcylist // bcc
    # Colours it in inside the mod image
    for y_change in range(-10, 11):
        for x_change in range(-10, 11):     
            mod[(bcy + y_change, bcx + x_change)] = [247, 127, 79]

    ######################
    ### GENERATE GREEN ###
    ######################
    temp = img.copy() # temporary image where all the white pixels are set to black
    black_pixels = np.where(
        (img[:, :, 0] >= 150) & 
        (img[:, :, 1] >= 150) & 
        (img[:, :, 2] >= 150) 
    )
    temp[black_pixels] = [0, 0, 0]
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) # Grayscales image 
    ret,template = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) # Forces a binary image in the opposite direction where black becomes white, green becomes black
    contours,hierarchy = cv2.findContours(template,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Find green contours
    green_squares = 0 # Number of green squares found
    gcs = [] # Green centroids
    for contour in contours[1:]: # Skip first centroid
        if cv2.contourArea(contour) > 5000: # Size requirement for it to be valid
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(contourimg, (x, y), (x + w, y + h),(0, 0, 255),2)
            gcs.append([x + w//2, y + h//2])
            #cv2.rectangle(mod,(x + w//2 - 5, y + h//2 - 5),(x + w//2 + 5,y + h//2 + 5), (0, 0, 255), -1)
            green_squares += 1

    #####################
    ### TURNING LOGIC ###
    #####################
    if green_squares: # If there is green found, there must be turning logic
        for gc in gcs:
            gcx,gcy = gc[0],gc[1]
            for x_change in range(-10, 11):
                for y_change in range(-10, 11):
                    mod[(gcy + y_change, gcx + x_change)] = [0, 0, 255] # Colours in the green square centroids
        green_squares_present = [False] * 4 # A list of the green squares which are present at the intersection(s) (Index 0 is top right, 1 is top left, 2 is bottom left, 3 is bottom right)
        green_squares_coords = [[0, 0]] * 4
        for gc in gcs:
            # Loops through every green sqaure
            gx, gy = gc[0], gc[1] # Gets the x and y values of the centroid of the green contour
            bgx = bgxc = bgy = bgyc = 0 # The black lines that border the green contour - this section finds their centroid
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
            # Colours the two black centroids and the lines that connect the centroids together in
            for x_change in range(-10, 11):
                for y_change in range(-10, 11):
                    mod[(gy + y_change, bgx + x_change)] = [0, 0, 255]
                    mod[(bgy + y_change, gx + x_change)] = [0, 0, 255]
            cv2.rectangle(mod,(min(gx,bgx),gy),(max(gx,bgx),gy),(0,0,255),10)
            cv2.rectangle(mod,(gx,min(gy,bgy)),(gx,max(gy,bgy)),(0,0,255),10)
            if gy > bgy:
                if gx > bgx: 
                    # If the green centroid's x is larger than the black line centroid's x, and the y is larger then it is a square on the bottom right of intersections
                    green_squares_present[3] = True
                    green_squares_coords[3] = [gx,gy]
                else: 
                    # If the green centroid's x is smaller than the black line centroid's x, and the y is larger then it is a square on the bottom left of intersections
                    green_squares_present[2] = True
                    green_squares_coords[2] = [gx,gy]
            else:
                if gx > bgx: 
                    # If the green centroid's x is larger than the black line centroid's x, and the y is smaller then it is a square on the top right of intersections
                    green_squares_present[0] = True
                    green_squares_coords[0] = [gx,gy]
                else: 
                    # If the green centroid's x is smaller than the black line centroid's x, and the y is smaller then it is a square on the top left of intersections
                    green_squares_present[1] = True
                    green_squares_coords[1] = [gx,gy]
        if green_squares_present[3] and green_squares_present[2]:
            # If both the bottom left and right squares are present, the choice is a U-turn
            decision = "Decision: U-Turn"
            to_return = [-1, -1]
        elif green_squares_present[3]:
            # If only the bottom right square is present, the choice is a right turn
            decision = "Decision: Turn Right"
            to_return = green_squares_coords[3]
        elif green_squares_present[2]:
            # If only the bottom left square is present, the choice is a left turn
            decision = "Decision: Turn Left"
            to_return = green_squares_coords[2]
        else:
            # Any other permutations means that the bot should continue forward
            decision = "Decision: Continue Forward" 
            to_return = [bcx,bcy]

        ### PLOTTING ###
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Contrast Image')
        plt.xlabel(decision)

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
        plt.savefig(path + '/analysed/' + filepath)
        plt.show()
        return to_return
    else:
        decision = "Decision: Follow blue dot"
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 5, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Contrast Image')

        plt.subplot(1, 5, 3)
        plt.imshow(cv2.cvtColor(btemplate, cv2.COLOR_BGR2RGB))
        plt.title('Binary Image')

        plt.subplot(1, 5, 4)
        plt.imshow(cv2.cvtColor(contourimg, cv2.COLOR_BGR2RGB))
        plt.title('Contours Image')

        plt.subplot(1, 5, 5)
        plt.imshow(cv2.cvtColor(mod, cv2.COLOR_BGR2RGB))
        plt.title('Analysed Image')
        plt.savefig(path + '/analysed/' + filepath)
        plt.show()
        return [bcx,bcy]
print(analyse_image(cv2.imread(filepath)))