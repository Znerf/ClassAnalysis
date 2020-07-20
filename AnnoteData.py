from modules.yolo import YoloDark

import cv2
import numpy as np

# from PIL import Image, ImageOps

def main():
    yolo= YoloDark()
    data= "annotateInput/data1.mp4"
    net= yolo.yoloInit()
    cap = cv2.VideoCapture(data)

    while True:
        has_frame, frame = cap.read()
        faces= yolo.yoloProcess(net,frame)
        
        img_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #set a thresh
        thresh = 100
        #get threshold image
        ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
        #find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #create an empty image for contours
        img_contours = np.zeros(frame.shape)
        # draw the contours on the empty image
        cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

        cv2.imshow("Cam", img_contours)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('Interrupted by user!')
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
