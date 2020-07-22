import cv2
from PIL import Image, ImageOps
from numpy import asarray

import sys
import os
import numpy as np


from modules.yolo import YoloDark
from modules.facenet import Facenet



def _main():
	faceRecog= Facenet()
	model=faceRecog.genModel()
	
	yolo= YoloDark()
	net= yolo.yoloInit()

	cap = cv2.VideoCapture(faceRecog.port)
	while True:

		has_frame, frame = cap.read()
		original=frame
        
		if not has_frame:
			
			cv2.waitKey(1000)
			break
		faces= yolo.yoloProcess(net,frame)

		pixels=asarray(frame)
		
		for face in faces:

			x1=face[0]
			y1=face[1]
			x2=face[0]+face[2]
			y2=face[1]+face[3]

			face = pixels[y1:y2, x1:x2]
			image = Image.fromarray(face)
			image = image.resize((160,160))
			image = np.asarray(image)


			predict_name,class_probability=faceRecog.process(model,image)
			print(class_probability)
			print(predict_name)
			try:
				frame = cv2.putText(frame, predict_name[0], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,255), 1, cv2.LINE_AA)
			except:
				pass

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
