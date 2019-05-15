# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

# GPU config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)


import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from glob import glob
import matplotlib.pyplot as plt

import os
from os import listdir, getcwd
from os.path import join
import shutil
import cv2
import numpy as np

class YOLO(object):
    _defaults = {
        "model_path": 'Container/weight/trained_weights_stage_1.h5',
        "anchors_path": 'Container/container_anchors.txt',
        "classes_path": 'Container/container_classes.txt',
        "score" : 0.1,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            #map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            map(lambda x: (int(x[0]), int(x[1]), int(x[2])),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,fn):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })        
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #thickness = (image.size[0] + image.size[1]) // 300

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.imshow(np.array(image))
        plt.axis('off')
        ax = plt.axes()
        ax.tick_params(labelbottom='off', bottom='off', labelleft='off', left='off')
        
        cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #draw = ImageDraw.Draw(image)
            #label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            lines = [fn ,c,left,top,abs((right-left)/image.size[1]),abs((bottom-top)/image.size[0]) ,score]
            global submission

            [submission.write(str(r) + ",") for r in lines]
            submission.write("\n")


            # PLT方式
            if score > 0.45:
                ax.add_patch(
                    plt.Rectangle((left, top), right-left, bottom-top,fill=False,
                                edgecolor=self.colors[c], linewidth=1.))

                ax.text(left+7, top, label,
                    bbox=dict(facecolor=self.colors[c], alpha=0.3, edgecolor=None), size=10, color='white')

            # CV2方式
            cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 0,255), 1)
            cv2.putText(cv_img, label, (left+7, top-8),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5, color=(0, 0, 255), thickness=1)

            
            #if top - label_size[1] >= 0:
            #    text_origin = np.array([left, top - label_size[1]])
            #else:
            #    text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            #for i in range(thickness):
            #    draw.rectangle(
            #        [left + i, top + i, right - i, bottom - i],
            #        outline=self.colors[c])
            #draw.rectangle(
            #    [tuple(text_origin), tuple(text_origin + label_size)],
            #    fill=self.colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            #del draw        
        end = timer()
        print(end - start)
        return cv_img,len(out_boxes)

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def detect(yolo,preety=True):
    out_dirs = 'Container/DETECTImages/'
    if not os.path.exists(out_dirs):
        os.makedirs(out_dirs)

    # lists = "Container/JPEGImages/"
    lists = "Container/test_pub_cdc/test_pub_images/"

    global submission
    submission = open("Container/Submission.csv","w") 
    submission.write('image_filename,label_id,x,y,w,h,confidence')
    
    for idx,f in enumerate(listdir(lists)):
        img_path = lists + f
        image = Image.open(img_path)
        # image = cv2.imread(img_path)
        r_image,bboxs = yolo.detect_image(image,f)
        # img = np.array(r_image)  
        out_path = out_dirs + f
        if bboxs>0:  
            if preety:       
                plt.savefig(out_dirs + "_" + f,dpi=300)   
            else:   
                cv2.imwrite(out_path,r_image) 
        submission.flush()

if __name__ == '__main__':
    detect(YOLO(),False)  