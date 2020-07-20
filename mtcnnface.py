

import cv2
from pretrained.mtcnna import MTCNN

def main():

    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    i=0
    while True:
        has_frame, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)

        cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

        cv2.imshow("Cam", image)

        # print(result)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
