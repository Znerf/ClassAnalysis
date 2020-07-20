# ClassAnalysis


YoloEmotion.py
It is test for Yolo face detection and emotion analysis. All pretrained codes are located at pretrained/emotion and yolo. It's main modules lies in modules/emotion.py and yolo.py

haartest.py
It include haarcascade code. It will send images to be processed. Uncomment the code there are comment rest to work it as harcascade face detection and cropping.

Haar cascade documentation

a=Haar("sagar",0)
a.reset("sagar",0)
a.run()


some static function
    directorycheck(name)            ->returns true if directory is not there and create directiory
    cropimage(img, x,y,w,h) :       ->returns true, image or false , status
    saveimage(filename, image):     ->returns true or false
    Haar(name,port)                 -> initialize haar and set name and port of camera


Note::


mtcnnface.py
It is library for mtcnn face detection with eye nose and mouth detection with points. It has a error on bounding box creation right now. so MTCNN is not finished use mtcnn library pip install mtcnn and you can use it directly
