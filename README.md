# ClassAnalysis


YoloEmotion.py<br/>
It is test for Yolo face detection and emotion analysis. All pretrained codes are located at pretrained/emotion and yolo. It's main modules lies in modules/emotion.py and yolo.py<br/>

haartest.py<br/>
It include haarcascade code. It will send images to be processed. Uncomment the code there are comment rest to work it as harcascade face detection and cropping.<br/>

Haar cascade documentation<br/>

a=Haar("sagar",0)<br/>
a.reset("sagar",0)<br/>
a.run()<br/>


some static function<br/>
    directorycheck(name)            ->returns true if directory is not there and create directiory<br/>
    cropimage(img, x,y,w,h) :       ->returns true, image or false , status<br/>
    saveimage(filename, image):     ->returns true or false<br/>
    Haar(name,port)                 -> initialize haar and set name and port of camera<br/>


Note::<br/>


mtcnnface.py<br/>
It is library for mtcnn face detection with eye nose and mouth detection with points. It is imported from modules.mtcnn. It has a error on bounding box creation right now. so MTCNN is not finished use mtcnn library pip install mtcnn and you can use it directly.<br/>


facerecog.py<br/>
This program doesnot have training program. It has to be added first . In this modules, it has imported modules/facenet.py and it file located at dataset/facerecog for datafile and embeddings. Also, pretrained/facenet


File strucutre<br/>
1. dataset  --> All the files that are fed to NN <br/>
2. documentation --> document of this project as well as ouput screen shots<br/>

3. 
