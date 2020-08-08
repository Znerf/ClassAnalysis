import cv2
from PIL import Image, ImageOps
from numpy import asarray

import sys
import os
import numpy as np
import time


from modules.yolo import YoloDark
from modules.facenet import Facenet
from modules.annotate import Annotate
	


def _main():
	faceRecog= Facenet(port="annotateData/input/data1/sample.mp4")
	model=faceRecog.genModel()
	
	yolo= YoloDark(boundingbox=False)
	net= yolo.yoloInit()

	cap = cv2.VideoCapture(faceRecog.port)
	# cap.set(cv2.CAP_PROP_FPS,1)
	frames=0
	t0 = time.time()    

	a=Annotate()
	
	while True:

		has_frame, frame = cap.read()
		original=frame
		# print("hasfram", has_frame)
		frames=frames+1
		videotime=float(frames) / float(cap.get(cv2.CAP_PROP_FPS))
		print("time in sec::",videotime)
		print("time spent::", time.time()-t0)

		
		if not has_frame:
			cv2.waitKey(1000)
			break

		if (frames % cap.get(cv2.CAP_PROP_FPS)==0):

			faces= yolo.yoloProcess(net,original)

			pixels=asarray(frame)
			
			for face in faces:

				x1=face[0]
				y1=face[1]
				x2=face[0]+face[2]
				y2=face[1]+face[3]
				
				try:
					face = pixels[y1:y2, x1:x2]
					image = Image.fromarray(face)
					image = image.resize((160,160))
					image = np.asarray(image)
					

					predict_name,class_probability=faceRecog.process(model,image)
					print(class_probability)
					print(predict_name)
					
					location='annotateData/output/data1/'+predict_name[0]+'/'
					a.directorycheck('annotateData')
					a.directorycheck('annotateData/output/')
					a.directorycheck('annotateData/output/data1/')
					a.directorycheck(location)
					# if(float(  videotime-float(int(videotime))>0.9  )):

				

				
					a.saveImage(str(int(videotime))+'.jpg',image,location)

					frame = cv2.putText(frame, predict_name[0], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,255), 1, cv2.LINE_AA)
				except:
					pass
			a.directorycheck('annotateData/output/data1/')
			a.directorycheck('annotateData/output/data1/frame/')
			a.saveImage(str(int(videotime))+'.jpg',frame,'annotateData/output/data1/frame/')



			

		cv2.imshow("Facerecog", original)
		key = cv2.waitKey(1)
		if key == 27 or key == ord('q'):
			print('[i] ==> Interrupted by user!')
			break

	cap.release()
	cv2.destroyAllWindows()

	print('==> All done!')


if __name__ == '__main__':
	_main()
