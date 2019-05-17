#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


#%%
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import os
from os import listdir
import pickle
import pandas as pd
import sys
import time

import face_recognition

# % matplotlib inline
import os
from os import listdir,getcwd


#%% 臉部識別函數宣告
#讀取評分數據
rating_dict={}
with open('./All_labels.txt','rb') as label:
    datalines = label.readlines()
    for d in datalines:
        d = str(d).replace('b','').replace('\\n','').replace('\\r','').replace("'","").split(' ')
        rating_dict[d[0]] = float(d[1])        

def get_face(fullpath,pred=True):
    img = face_recognition.load_image_file(fullpath)
    # (Top,Left,Buttom,right)
    locat = face_recognition.face_locations(img,model="cnn")
    if len(locat)>0:
        Top,right,Buttom,Left = locat[0]    
        img = img[Top:Buttom, Left:right] 
        img = cv2.resize(img, (128, 128))         
        return img 
    else:
        return np.array([0]) 
    

#%% 顏值特徵擷取
def load_dataset():  
    files = './Images'
    image_data_list = []
    label = []
    start = time.time()
    # 以迴圈處理
    error = []
    for idx,f in enumerate(os.listdir(files)):
        # 產生檔案的絕對路徑
        fullpath = os.path.join(files, f)
        # img = cv2.imread(fullpath)
        face = get_face(fullpath)
        if (face.shape != (1,)) :
            image_data_list.append(img_to_array(face))
            label.append(rating_dict[f]) 
        else:
            error.append(f)
            print(fullpath)
        if (idx%100==0)and (idx>0):        
            print("{} detect success , use time:{:2f}s".format(idx - len(error),time.time() - start))
        del fullpath,face
        
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255        
    return img_data, label 



#%% DataSet輸出為npz檔
if not os.path.exists('./Image.npz'):
    train_x, train_y = load_dataset()
    train_x.shape
else:
    ds = np.load('./Image.npz')
    train_x,train_y = ds['data'],ds['label']
    len(rating_dict.keys())


np.savez('./Image.npz', data=train_x, label=train_y)
ds = np.load('./Image.npz')
train_x,train_y = ds['data'],ds['label']
train_y = np.array(train_y)
len(train_y)

#%% 查看顏值分布情況
scores = list(train_y)
lv1 = [x for x in scores if x<1]
lv2 = [x for x in scores if x>=1 and x<1.5]
lv3 = [x for x in scores if x>=1.5 and x<2]
lv4 = [x for x in scores if x>=2 and x<2.5]
lv5 = [x for x in scores if x>=2.5 and x<3]
lv6 = [x for x in scores if x>=3 and x<3.5]
lv7 = [x for x in scores if x>=3.5 and x<4]
lv8 = [x for x in scores if x>=4 and x<4.5]
lv9 = [x for x in scores if x>=4.5]
plt.bar(['lv1','lv2','lv3','lv4','lv5','lv6','lv7','lv8','lv9'],
       [len(x) for x in [lv1,lv2,lv3,lv4,lv5,lv6,lv7,lv8,lv9]])

#%% 建立顏值評分模型
# def make_network():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
#     model.add(Activation('relu'))
# #     model.add(Conv2D(32, (3, 3)))
# #     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))
    
#     model.add(Flatten())
#     model.add(BatchNormalization())
# #     model.add(Dense(1024, activation='relu'))
# #     model.add(Dropout(0.3))
# #     model.add(BatchNormalization())
#     model.add(Dense(1))
# #     model.add(Activation('linear'))
#     model.compile(loss='mean_squared_error', optimizer='Adam',metrics=['accuracy'])
#     model.summary()
#     return model
#%%
seed = 1
x_train_all, x_test, y_train_all, y_test = train_test_split(train_x, np.array(train_y), test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=seed)
#%%
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# # config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=config)
# KTF.set_session(sess)
#%%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def make_network():
    resnet = ResNet50(include_top=False, pooling='avg', input_shape=(128, 128, 3))
    model = Sequential()
    model.add(resnet)
    model.add(Dense(1))
    model.layers[0].trainable = True
#     model.compile(loss='mse', optimizer='adam')    
    # model.compile(loss='mse', optimizer='Adam',metrics=['accuracy'])
    model.compile(loss = root_mean_squared_error, optimizer='Adam',metrics=['accuracy'])
    model.summary()
    model.summary()
    return model

model = make_network()

#%% 模型訓練
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience = 10, 
    verbose = 0, 
    mode='auto'
)

filepath="{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callback_list = [checkpoint, reduce_learning_rate,early_stopping]

# callback_list = [checkpoint,early_stopping]

train_history = model.fit(x_train, y_train,                           
                          batch_size=8, epochs=50, verbose=1, 
                          validation_split=0.2,
                          validation_data=(x_val, y_val),
                          callbacks = callback_list)


#%% 模型評估
def show_train_history(train_history, train, validation,title):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.title(title)
    plt.show()
    
show_train_history(train_history, 'acc','val_acc','Accuracy')
show_train_history(train_history, 'loss','val_loss','Loss')
loss,acc = model.evaluate(x_test,y_test)
print("Loss:{} , Accuracy:{}".format(loss,acc))
#%%
plt.scatter(y_test,model.predict(x_test))
plt.plot(y_test,y_test,'ro')
#%% 儲存模型%權重
# Save model
model.save('face_model/faceRank.h5')
model.save_weights('face_model/faceRank_weights.h5')
del model

model = load_model('face_model/faceRank.h5',
 custom_objects={'root_mean_squared_error': root_mean_squared_error})
model.load_weights('face_model/faceRank_weights.h5')

#%% 目標顏值預測
def predict_image(img_url):
    try:
        image = cv2.imread(img_url)
        face = get_face(image)
        face = face.astype('float32')
        face /= 255  
        print(face.shape)
        image = img_to_array(face)
        img = image[np.newaxis,:,:]
        plt.axis('off')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        

        print("Predict Score : {}".format(model.predict(img)[0][0] * 20))  
    except Exception as e :
        print(e)
        print('臉部辨識失敗')

#%%
files = "./Test/"
for idx,f in enumerate(listdir(files)):
    # 產生檔案的絕對路徑
    fullpath = os.path.join(files, f)
    face = get_face(fullpath)
    face = img.astype('float32')
    face /= 255  
    print(face.shape)
    image = img_to_array(face)
    img = image[np.newaxis,:,:]
    plt.axis('off')
    plt.figure()
    # plt.imshow(img)
    plt.show()
    

    print("Predict Score : {}".format(model.predict(img)[0][0] * 20))   

    # img = load_img(fullpath)
    # predict_image(fullpath)

#%%
