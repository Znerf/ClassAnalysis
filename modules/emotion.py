import numpy as np
import os
import sys
import keras
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

# from PIL import Image

from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Dense,Dropout,Flatten
from keras import Sequential
# from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


class Emotion():
    def __init__(self, train= False, csvlocation="dataset/fer2013/fer2013.csv"):

        config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
        self.classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.output="pretrained/emotion"
        self.num_classes=7
        self.batch_size = 256
        self.epochs = 5
        self.csvlocation=csvlocation

        if train is True:
            if (Emotion.directorycheck(self.output.split("/")[0])):
                Emotion.directorycheck(self.output.split("/")[0]+"/"+self.output.split("/")[1])

        else:
            self.model_json=self.output+"/model.json"
            self.model_weight=self.output+"/model.h5"
            if os.path.exists(self.model_weight):
                if os.path.exists(self.model_json):
                    pass
                else:
                    raise ValueError("No model found")
            else:
                raise ValueError("No weight found")
            # self.load_weights(model_json,model_weight)



    @staticmethod
    def directorycheck(name):
        if (not(os.path.exists(name))):
            os.mkdir(name)
            return True
        else:
            return False



    def readCsv(self):
        if (os.path.exists(self.csvlocation)):
            # print("Path")

            with open(self.csvlocation) as f:
                content = f.readlines()
                lines = np.array(content)
                # num_of_instances = lines.size #number of instances:  35888 in fer2013.csv
            return lines
        else:
            print("No valid CSV link given")
            sys.exit()

    def trainTestSplit(self,lines):
        x_train, y_train, x_test, y_test = [], [], [], []

        for i in range(1,lines.size):
            try:

                emotion, img, usage = lines[i].split(",")
                val = img.split(" ")
                pixels = np.array(val, 'float32')
                # pixels = np.asarray(val, dtype=np.uint8).reshape(48,48)
                emotion = keras.utils.to_categorical(emotion, self.num_classes)

                if 'Training' in usage:
                    y_train.append(emotion)
                    x_train.append(pixels)
                elif 'PublicTest' in usage:
                    y_test.append(emotion)
                    x_test.append(pixels)

            except:

                print("Train test Spliting Error", end="")
                sys.exit()
        # print(len(x_train))#28709
        # print(len(y_train))#
        # print(len(x_test))#3589
        # print(len(x_test))#

        x_train = np.array(x_train, 'float32')
        y_train = np.array(y_train, 'float32')
        x_test = np.array(x_test, 'float32')
        y_test = np.array(y_test, 'float32')

        x_train /= 255 #normalize inputs between [0, 1]
        x_test /= 255

        x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test


    def arch(self):
        model = Sequential()

        #1st convolution layer
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

        #2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        #3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        model.add(Flatten())

        #fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def compile(self, x_train,y_train, model, batch_size=1,epochs=1):
        gen = ImageDataGenerator()
        train_generator = gen.flow(x_train, y_train, batch_size=self.batch_size, seed= None)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        model.fit_generator(train_generator, steps_per_epoch= len(x_train) / self.batch_size, epochs=self.epochs)
        self.writeModelToFile(model)
        return model

    def run(self):  # Train main
        csv= self.readCsv()
        x_train, y_train, x_test, y_test = self.trainTestSplit(csv)
        model= self.arch()
        model= self.compile(x_train,y_train, model)
        self.evaluate(model,x_test, y_test)
        print("Success")


    def writeModelToFile(self,model):
        model_json = model.to_json()
        with open(self.output+"/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(self.output+"/model.h5")
        print("Saved model to disk")

    def checkrun(self):     # Test main
        model= self.load_weights()
        # self.confusion(model)
        img = image.load_img("test.png",color_mode = "grayscale", target_size=(48, 48))
        _,index= self.checkout(model,img)

        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis = 0)
        #
        # x /= 255
        # x = np.array(x, 'float32')
        # x = x.reshape([48, 48]);
        # plt.gray()
        # plt.imshow(x)
        # plt.show()
        # print(index)

    def load_weights(self):

        json_file = open(self.model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(self.model_weight)
        print("Loaded model from disk")


        # evaluate loaded model on test data
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        # csv= self.readCsv()
        # _, _, x_test, y_test = self.trainTestSplit(csv)
        # self.evaluate(model,x_test, y_test)

        # model.load_weights('/data/facial_expression_model_weights.h5') #load weights
        return model
        # self.confusion(model)

    def confusion(self,model):
        csv= self.readCsv()
        _, _, x_test, y_test = self.trainTestSplit(csv)
        # self.evaluate(model,x_test, y_test)

        predictions = model.predict(x_test)
        out=[]
        for a in predictions:
            out.append(np.argmax(a))
        act=[]
        for a in y_test:
            act.append(np.argmax(a))

        conMat= confusion_matrix(act, out)
        print(conMat)
        print(len(act))
        print(len(out))
        print(len(y_test))

    def checkout(self,model,img, show=False):

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)

        x /= 255
        # x=img
        custom = model.predict(x)
        if show is True:
            self.emotion_analysis(custom[0])
        return custom[0], np.argmax(custom[0])

    def testcheck(self,model):
        csv= self.readCsv()
        _, _, x_test, y_test = self.trainTestSplit(csv)
        # self.evaluate(model,x_test, y_test)

        predictions = model.predict(x_test)
        index=0
        for i in predictions:
            if index < 30 and index >= 20:
                testing_img = np.array(x_test[index], 'float32')
                testing_img = testing_img.reshape([48, 48]);

                plt.gray()
                plt.imshow(testing_img)
                plt.show()

                print(i)

                self.emotion_analysis(i)
            index = index + 1

    def emotion_analysis(self,emotions):
        objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        y_pos = np.arange(len(objects))

        plt.bar(y_pos, emotions, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('percentage')
        plt.title('emotion')

        plt.show()
    def evaluate(self,model,x_test,y_test):
        score = model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', 100*score[1])

"""
a=Haar("sagar",0)
a.reset("sagar",0)
a.run()

Haar cascade documentation
some static function
    directorycheck(name)            ->returns true if directory is not there and create directiory
    cropimage(img, x,y,w,h) :       ->returns true, image or false , status
    saveimage(filename, image):     ->returns true or false
    Haar(name,port)                 -> initialize haar and set name and port of camera

"""
