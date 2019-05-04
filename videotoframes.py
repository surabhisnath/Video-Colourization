import cv2
cnt = 10
vidcap = cv2.VideoCapture("./videos/" + str(cnt) + ".mp4")
success,image = vidcap.read(0)
count = 0
while success:
	if(count%12 == 0):	    
		cv2.imwrite("./grayframes/" + str(cnt) + "_frames/frame" + str(count) + ".jpg", image)    # save frame as JPEG file  
		print('Read a new frame: ', success)
	success,image = vidcap.read()
	count += 1