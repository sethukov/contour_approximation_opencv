import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Location to image file
input=sys.argv[1]
img = cv2.imread(input,0)
ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

zeros_image=np.zeros(im2.shape)
# https://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#void%20line(Mat&%20img,%20Point%20pt1,%20Point%20pt2,%20const%20Scalar&%20color,%20int%20thickness,%20int%20lineType,%20int%20shift)
# https://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html

# Method 1
useable_points=[]
for num in range(len(approx)):
    useable_points.append((approx[num][0][0],approx[num][0][1]))
useable_points = np.array(useable_points, np.int32)
useable_points = useable_points.reshape((-1,1,2))
img_approximation=cv2.polylines(0*zeros_image.copy(),[useable_points],True,(255,255,255),1,1)
plt.subplot('121')
plt.title("Method 1: Using cv2.polylines")
plt.imshow(img_approximation)
#plt.show()

# Method 2
# Drawing point by points
# Convoluted method

for num in range(len(approx)):

    if num!=len(approx)-1:
        zeros_image=cv2.line(zeros_image, (approx[num][0][0],approx[num][0][1]), (approx[num+1][0][0],approx[num+1][0][1]), 255, 1, 1)
    else:
        zeros_image=cv2.line(zeros_image, (approx[num][0][0],approx[num][0][1]), (approx[0][0][0],approx[0][0][1]), 255, 1, 1)

plt.subplot('122')
plt.title("Method 2: Using cv2.line")
plt.imshow(zeros_image)
plt.show()
