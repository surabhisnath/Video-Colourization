import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps, folder):
	frame_array = []
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
	#for sorting the file names properly
	
	files.sort(key = lambda x: int(x[5:-4]))
 
	for i in range(len(files)):
		filename=pathIn + files[i]
		#reading each files
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		print(filename)
		#inserting the frames into an image array
		frame_array.append(img)
 
	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
 
	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()
 
def main():
	cnt = 1
	files = os.listdir("./grayframes/")
	files.sort(key = lambda x: int(x[:-7]))

	for folder in files:
		# if folder != "10_frames":
		# 	continue
		print(folder)
		pathIn= "./grayframes/" + folder + "/"
		pathOut = "./grayvideos/" + str(cnt) + ".mp4"
		fps = 3
		convert_frames_to_video(pathIn, pathOut, fps, folder)
		cnt += 1
 
if __name__=="__main__":
	main()
