import cv2
import os

face_cascade = cv2.CascadeClassifier('pretrained/haar/frontal.xml')
eye_cascade = cv2.CascadeClassifier('pretrained/haar/eye.xml')

class Haar():
    def __init__(self, name,port):

        self.i=0

        self.directorycheck("dataset")
        self.directorycheck("dataset/HaarData")
        if self.directorycheck("dataset/HaarData/"+name):
            self.status="Success"
        else:
            self.status="Override Data"

        self.name=name;
        self.port=port
        self.directorycheck(name)

    def reset(self,name,port):
        self.name=name
        self.i=0
        if self.directorycheck("dataset/HaarData/"+name):
            self.status="Success"
        else:
            self.status="Override Data"
        self.port=port

    @staticmethod
    def directorycheck(name):
        if (not(os.path.exists(name))):
            os.mkdir(name)
            return True
        else:
            return False

    @staticmethod
    def cropimage(img, x,y,w,h) :
        try:
            return True, img[y:y+h, x:x+w]
        except:
            return False, "cropping error"

    @staticmethod
    def saveimage(filename, image):
        try:
            cv2.imwrite( filename, image );
            return True
        except:
            return False


    def haarcas(self,img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return face_cascade.detectMultiScale(gray, 1.3, 5)
        # print (faces)



    def run(self):
        video_capture = cv2.VideoCapture(self.port)

        while True:
            ret, frame = video_capture.read()
            if frame is None:
                continue

            faces= self.haarcas(frame)

            for (x,y,w,h) in faces:
                check, crop_img=self.cropimage(frame,x,y,w,h)
                if check is False:
                    # self.status=crop_img
                    continue

                filename="dataset/HaarData/"+self.name+"/"+str(self.i)+".jpg"
                if self.saveimage(filename, crop_img) is False:
                    continue

                self.i=self.i+1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)
        video_capture.release()
        cv2.destroyAllWindows()


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
