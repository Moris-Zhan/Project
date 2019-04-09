from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
import numpy as np
import cv2
import dlib

import dlib
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import imutils
from imutils.face_utils import *
import os
from os import listdir
import pickle
import pandas as pd
import sys
import time

from keras.callbacks import *

        

def get_face(img):
    #產生臉部識別
    face_rects = detector(img, 1)
    for i, d in enumerate(face_rects):
        #讀取框左上右下座標
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        #根據此座標範圍讀取臉部特徵點
        shape = landmark_predictor(img, d)
        #將特徵點轉為numpy
        shape = shape_to_np(shape)# (68,2)    
        # 透過dlib挖取臉孔部分，將臉孔圖片縮放至128*128的大小，並存放於npz檔中
        # 人臉圖像部分呢。很簡單，只要根據畫框的位置切取即可crop_img = img[y1:y2, x1:x2, :]
        crop_img = img[y1:y2, x1:x2, :]   
        try:
            crop_img = cv2.resize(crop_img, (128, 128))         
            return crop_img   
        except:
            return np.array([0])  
    return np.array([0]) 
        



def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())    
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss ='mse',optimizer='adam')
    model.summary()
    return model

def load_image(img_url):
    image = load_img(img_url,target_size=(128,128))
    image = img_to_array(image)
    image /= 255
    image = np.expand_dims(image,axis=0)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def main():
    global rating_dict
    rating_dict={}
    with open('./beauty_detector/All_labels.txt','rb') as label:
        datalines = label.readlines()
        for d in datalines:
            d = str(d).replace('b','').replace('\\n','').replace("'","").split(' ')
            rating_dict[d[0]] = float(d[1])

    train_x, train_y = load_dataset()
    np.savez('./beauty_detector/my_archive.npz', data=train_x, label=train_y)
    ds = np.load('./beauty_detector/my_archive.npz')
    train_x,train_y = ds['data'],ds['label']
    model = make_network()

    # Training model
    early_stopping = EarlyStopping(monitor='val_loss',patience=2,verbose=0,mode='auto')
    hist = model.fit(train_x, np.array(train_y), batch_size=32, epochs=1, verbose=1, callbacks=[early_stopping])    
    model.evaluate(train_x,train_y)
    # Save model
    model.save('./beauty_detector/faceRank.h5')
    model.save_weights('./beauty_detector/faceRank_weights.h5')
    del model
    from keras.models import load_model
    model = load_model('./beauty_detector/faceRank.h5')
    model.load_weights('./beauty_detector/faceRank_weights.h5')

def load_dataset():
    global detector ,landmark_predictor
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('./beauty_detector/shape_predictor_68_face_landmarks.dat')
    files = './beauty_detector/Images'
    image_data_list = []
    label = []
    error = []
    start = time.time()
    for idx,f in enumerate(os.listdir(files)):
        # 產生檔案的絕對路徑
        fullpath = os.path.join(files, f)
        img = cv2.imread(fullpath)
        face = get_face(img)
        if (face.shape != (1,)) :
            image_data_list.append(img_to_array(face))
            label.append(rating_dict[f]) 
        else:
            error.append(f)
            print(fullpath)

        if (idx%100==0)and (idx>0):        
            print("{} detect success , use time:{:2f}s".format(idx - len(error),time.time() - start))
        del fullpath,img,face
            
    print(len(error))
    print(error)
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255        
    return img_data, label 

main()