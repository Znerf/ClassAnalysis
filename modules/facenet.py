from numpy import load

from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

from keras.models import load_model


import os
import numpy as np


class Facenet():
    def __init__(self, train= False):

        self.model='pretrained/facenet/facenet_keras.h5'
        self.port='./input/recog/sample.mp4'

        if os.path.exists(self.model):
            pass
        else:
            raise ValueError("No model found")

        self.Facenet_model = load_model(self.model)
        self.out_encoder = LabelEncoder()
        self.dataset='dataset/faceRecog/faces-dataset.npz'
        self.embedding='dataset/faceRecog/embeddings.npz'

    def genModel(self):
        data = load(self.dataset)
        testX_faces = data['arr_2']
        # load face embeddings
        data = load(self.embedding)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print(trainX.shape)
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        
        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        testy = self.out_encoder.transform(testy)
        # fit model
        model = SVC(kernel='linear', probability=True)
        return model.fit(trainX, trainy)

    def process(self,model,image):  # Train main
        embedding = self.get_embedding( image)
        samples = expand_dims(embedding, axis=0)

        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index]*100
        predict_name = self.out_encoder.inverse_transform(yhat_class)

        if class_probability<99.5:
            predict_name[0]="Donot know"

        return predict_name,class_probability

    def get_embedding(self, face_pixels):
        model=self.Facenet_model
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
