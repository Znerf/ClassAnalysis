from numpy import load
from numpy import asarray
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

import tensorflow as tf
from sklearn.svm import SVC
from matplotlib import pyplot
import cv2
from PIL import Image, ImageOps

from keras.models import load_model

import argparse
import sys
import os
import numpy as np


from modules.yolo import YoloDark




def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	# print("Embedding /n /n")
	return yhat[0]



Facenet_model = load_model('pretrained/facenet/facenet_keras.h5')


video='./input/recog/sample.mp4'
# output_dir= 'dataset/facerecog/'

data = load('dataset/faceRecog/faces-dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('dataset/faceRecog/embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(trainX.shape)
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)



def _main():
	wind_name = 'face detection using YOLOv3'

	if not os.path.isfile(video):
		print("[!] ==> Input video file {} doesn't exist".format(video))
		sys.exit(1)
	cap = cv2.VideoCapture(video)

	yolo= YoloDark()
	net= yolo.yoloInit()

    
	while True:

		has_frame, frame = cap.read()
		original=frame
        # Stop the program if reached end of video
		if not has_frame:
			print('[i] ==> Done processing!!!')
            # print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
			cv2.waitKey(1000)
			break
		faces= yolo.yoloProcess(net,frame)

		pixels=asarray(frame)
		# print (faces)
		for face in faces:

			x1=face[0]
			y1=face[1]
			x2=face[0]+face[2]
			y2=face[1]+face[3]

			face = pixels[y1:y2, x1:x2]
			image = Image.fromarray(face)
			image = image.resize((160,160))
			image = np.asarray(image)


            # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),6)
			embedding = get_embedding(Facenet_model, image)

			samples = expand_dims(embedding, axis=0)
			yhat_class = model.predict(samples)
			yhat_prob = model.predict_proba(samples)
			class_index = yhat_class[0]
			class_probability = yhat_prob[0,class_index]*100
			predict_name = out_encoder.inverse_transform(yhat_class)
			if class_probability<99.5:
				predict_name[0]="Donot know"
			print(class_probability)
			print(predict_name)
			try:
				frame = cv2.putText(frame, predict_name[0] , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,255), 1, cv2.LINE_AA)
                # pass
			except:
				pass

		cv2.imshow(wind_name, original)
		key = cv2.waitKey(1)
		if key == 27 or key == ord('q'):
			print('[i] ==> Interrupted by user!')
			break

	cap.release()
	cv2.destroyAllWindows()

	print('==> All done!')
	print('***********************************************************')


if __name__ == '__main__':
	_main()
