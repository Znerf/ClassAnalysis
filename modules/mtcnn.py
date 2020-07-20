import cv2
import os
from modules.Haar import Haar


class Mtcnn():
    def __init__(self):

        self.i=0
        self.src="/dataset/HaarData/"
        self.out="/dataset/MtcnnData/"
        if (Haar.directorycheck("dataset/HaarData")):
            assert("No data")

        Haar.directorycheck("dataset/MtcnnData")



    def to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def get_dataset(path, has_class_directories=True):
        pass
        dataset = []
        path_exp = os.path.expanduser(path)
        classes = [path for path in os.listdir(path_exp) \
                        if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        print("Reading datasets Path-----")
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
            print(dataset[i])
        return dataset

    def store_revision_info(src_path, output_dir, arg_string):
        try:
            # Get git hash
            cmd = ['git', 'rev-parse',"HEAD"]
            gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
            (stdout, _) = gitproc.communicate()
            git_hash = stdout.strip()

        except OSError as e:
            git_hash = ' '.join(cmd) + ': ' +  e.strerror


        try:
            # Get local changes
            cmd = ['git', 'diff', 'HEAD']
            gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
            (stdout, _) = gitproc.communicate()
            git_diff = stdout.strip()
        except OSError as e:
            git_diff = ' '.join(cmd) + ': ' +  e.strerror

        # Store a text file in the log directory
        rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
        # print(output_dir)
        with open(rev_info_filename, "w") as text_file:
            text_file.write('arguments: %s\n--------------------\n' % arg_string)
            text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
            text_file.write('git hash: %s\n--------------------\n' % git_hash)
            text_file.write('%s' % git_diff)

    def reset(self,name,port):
        self.name=name
        self.i=0
        if self.directorycheck("dataset/MtcnnData/"+name):
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

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print (faces)
        for (x,y,w,h) in faces:
            check, crop_img=self.cropimage(img,x,y,w,h)
            if check is False:
                # self.status=crop_img
                continue

            filename="dataset/MtcnnData/"+self.name+"/"+str(self.i)+".jpg"
            if self.saveimage(filename, crop_img) is False:
                continue

            self.i=self.i+1


    def run(self):
        video_capture = cv2.VideoCapture(self.port)

        while True:
            ret, frame = video_capture.read()
            if frame is None:
                continue

            self.haarcas(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)
        video_capture.release()
        cv2.destroyAllWindows()

'''


from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if args.detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    print(sys.argv)
'''
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
