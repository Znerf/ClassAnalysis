from random import choice
from numpy import load
from numpy import asarray
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from sklearn.svm import SVC
from matplotlib import pyplot
import cv2
from PIL import Image

from keras.models import load_model

import argparse
import sys
import os

from utils import *
from modules.yolo import YoloDark



def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss



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

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='./sample.mp4',
                     help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################
# print the arguments
# print('----- info -----')
# print('[i] The config file: ', args.model_cfg)
# print('[i] The weights of model file: ', args.model_weights)
# print('[i] Path to image file: ', args.image)
# print('[i] Path to video file: ', args.video)
# print('###########################################################\n')

data = load('dataset/facerecog/faces-dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('dataset/facerecog/embeddings.npz')
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


# Give the configuration and weight files for the model and load the network
# using them.
# net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main():
	wind_name = 'face detection using YOLOv3'
    # cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    # output_file = ''

	if args.image:
		if not os.path.isfile(args.image):
			print("[!] ==> Input image file {} doesn't exist".format(args.image))
			sys.exit(1)
		cap = cv2.VideoCapture(args.image)
		output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
	elif args.video:
		if not os.path.isfile(args.video):
			print("[!] ==> Input video file {} doesn't exist".format(args.video))
			sys.exit(1)
		cap = cv2.VideoCapture(args.video)
        # output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
	else:
        # Get data from the camera
		cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    # if not args.image:
    #     video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
    #                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    #                                    cap.get(cv2.CAP_PROP_FPS), (
    #                                        round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	while True:

		# has_frame, frame = cap.read()
		# original=frame
        # # Stop the program if reached end of video
		# if not has_frame:
		# 	print('[i] ==> Done processing!!!')
        #     # print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
		# 	cv2.waitKey(1000)
		# 	break
        #
        # # Create a 4D blob from a frame.
		# blob = cv2.dnn.blobFromImage(frame, 1/255 , (IMG_WIDTH, IMG_HEIGHT),
        #                              [0, 0, 0], 1, crop=False)
        #
        # # Sets the input to the network
		# net.setInput(blob)
        #
        # # Runs the forward pass to get output of the output layers
		# outs = net.forward(get_outputs_names(net))
        #
        # # Remove the bounding boxes with low confidence
		# faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
		# print('[i] ==> # detected faces: {}'.format(len(faces)))
		# print('#' * 60)

		pixels=asarray(frame)
		# print (faces)
		for face in faces:
			# facelist=list()
			# facelistfinal=list()
			x1=face[0]
			y1=face[1]
			x2=face[0]+face[2]
			y2=face[1]+face[3]
			#
			# crop = crop.convert('RGB')
			# crop=asarray(frame)
			# print(crop.shape)
			# crop=crop[y1:y2,x1:x2]
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
        # # initialize the set of information we'll displaying on the frame
        # info = [
        #     ('number of faces detected', '{}'.format(len(faces)))
        # ]
        #
        # for (i, (txt, val)) in enumerate(info):
        #     text = '{}: {}'.format(txt, val)
        #     cv2.putText(frame, text, (10, (i * 20) + 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        # if args.image:
        #     cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        # else:
        #     video_writer.write(frame.astype(np.uint8))
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
