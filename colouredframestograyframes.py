import cv2
import os
import numpy as np

cnt = 10
for img in os.listdir("./grayframes/" + str(cnt) + "_frames/"):
	print(img)
	im = cv2.imread("./grayframes/" + str(cnt) + "_frames/" + img, 0)
	cv2.imwrite("./grayframes/" + str(cnt) + "_frames/" + img, im)