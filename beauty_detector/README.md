

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.callbacks import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
```

    Using TensorFlow backend.
    


```python
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
```


```python
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
% matplotlib inline
```


```python
#%%
global detector ,landmark_predictor
#宣告臉部偵測器，以及載入預訓練的臉部特徵點模型
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./face_model/shape_predictor_68_face_landmarks.dat')

#%% 臉部識別函數宣告
#讀取評分數據
rating_dict={}
with open('./All_labels.txt','rb') as label:
    datalines = label.readlines()
    for d in datalines:
        d = str(d).replace('b','').replace('\\n','').replace("'","").split(' ')
        rating_dict[d[0]] = float(d[1])  
        
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
        # 透過dlib挖取臉孔部分，將臉孔圖片縮放至256*256的大小，並存放於pickle檔中
        # 人臉圖像部分呢。很簡單，只要根據畫框的位置切取即可crop_img = img[y1:y2, x1:x2, :]
        crop_img = img[y1:y2, x1:x2, :]   
        try:
            crop_img = cv2.resize(crop_img, (128, 128))         
            return crop_img   
        except:
            return np.array([0])  
    return np.array([0])         
```


```python
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
        
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255        
    return img_data, label 
```


```python
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
```




    5483




```python
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
lv = [1,2,3,4,5,6,7,8,9]
lv_label = ['lv1','lv2','lv3','lv4','lv5','lv6','lv7','lv8','lv9']
len_lv = [len(x) for x in [lv1,lv2,lv3,lv4,lv5,lv6,lv7,lv8,lv9]]
plt.bar(lv,len_lv)
plt.xticks(lv, lv_label)
```




    ([<matplotlib.axis.XTick at 0x23cb82cfc88>,
      <matplotlib.axis.XTick at 0x23cb82cffd0>,
      <matplotlib.axis.XTick at 0x23cb832e7b8>,
      <matplotlib.axis.XTick at 0x23cba372128>,
      <matplotlib.axis.XTick at 0x23cba372780>,
      <matplotlib.axis.XTick at 0x23cba372dd8>,
      <matplotlib.axis.XTick at 0x23cba3774a8>,
      <matplotlib.axis.XTick at 0x23cba377b38>,
      <matplotlib.axis.XTick at 0x23cba37d208>],
     <a list of 9 Text xticklabel objects>)




![png](output_6_1.png)



```python
#%%
seed = 42
x_train_all, x_test, y_train_all, y_test = train_test_split(train_x, np.array(train_y), test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=seed)

import tensorflow as tf
def correlation_coefficient(y_true, y_pred):
    pearson_r, _ = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)
    return 1-pearson_r**2

from scipy import stats

#%%
def make_network():
    resnet = ResNet50(include_top=False, pooling='avg', input_shape=(128, 128, 3))
    model = Sequential()
    model.add(resnet)
    model.add(Dense(1))
    model.layers[0].trainable = True
    model.compile(loss='mse', optimizer='Adam',metrics=['accuracy'])
#     model.compile(loss = correlation_coefficient, optimizer='Adam',metrics=['accuracy'])
#     model.compile(loss = stats.pearsonr, optimizer='Adam',metrics=['accuracy'])
    model.summary()
    return model
```


```python
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

train_history = model.fit(x_train, y_train, 
                          batch_size=8, epochs=30, verbose=1, 
                          validation_split=0.2,
                          validation_data=(x_val, y_val),
                          callbacks = callback_list)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    resnet50 (Model)             (None, 2048)              23587712  
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 2049      
    =================================================================
    Total params: 23,589,761
    Trainable params: 23,536,641
    Non-trainable params: 53,120
    _________________________________________________________________
    Train on 3508 samples, validate on 878 samples
    Epoch 1/30
    3508/3508 [==============================] - 102s 29ms/step - loss: 0.6633 - acc: 0.0080 - val_loss: 0.3688 - val_acc: 0.0068
    
    Epoch 00001: val_loss improved from inf to 0.36879, saving model to 01-0.37.h5
    Epoch 2/30
    3508/3508 [==============================] - 91s 26ms/step - loss: 0.2012 - acc: 0.0103 - val_loss: 0.2677 - val_acc: 0.0114
    
    Epoch 00002: val_loss improved from 0.36879 to 0.26772, saving model to 02-0.27.h5
    Epoch 3/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.1562 - acc: 0.0111 - val_loss: 0.1832 - val_acc: 0.0080
    
    Epoch 00003: val_loss improved from 0.26772 to 0.18320, saving model to 03-0.18.h5
    Epoch 4/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.1087 - acc: 0.0125 - val_loss: 0.1541 - val_acc: 0.0137
    
    Epoch 00004: val_loss improved from 0.18320 to 0.15410, saving model to 04-0.15.h5
    Epoch 5/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0882 - acc: 0.0145 - val_loss: 0.1456 - val_acc: 0.0114
    
    Epoch 00005: val_loss improved from 0.15410 to 0.14562, saving model to 05-0.15.h5
    Epoch 6/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0862 - acc: 0.0140 - val_loss: 0.2384 - val_acc: 0.0103
    
    Epoch 00006: val_loss did not improve from 0.14562
    Epoch 7/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0830 - acc: 0.0140 - val_loss: 0.2042 - val_acc: 0.0114
    
    Epoch 00007: val_loss did not improve from 0.14562
    Epoch 8/30
    3508/3508 [==============================] - 89s 26ms/step - loss: 0.0802 - acc: 0.0143 - val_loss: 0.1664 - val_acc: 0.0103
    
    Epoch 00008: val_loss did not improve from 0.14562
    Epoch 9/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0770 - acc: 0.0140 - val_loss: 0.3439 - val_acc: 0.0091
    
    Epoch 00009: val_loss did not improve from 0.14562
    Epoch 10/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0782 - acc: 0.0123 - val_loss: 0.1963 - val_acc: 0.0068
    
    Epoch 00010: val_loss did not improve from 0.14562
    Epoch 11/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0839 - acc: 0.0145 - val_loss: 0.2549 - val_acc: 0.0080
    
    Epoch 00011: val_loss did not improve from 0.14562
    
    Epoch 00011: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 12/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0401 - acc: 0.0145 - val_loss: 0.1178 - val_acc: 0.0137
    
    Epoch 00012: val_loss improved from 0.14562 to 0.11778, saving model to 12-0.12.h5
    Epoch 13/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0238 - acc: 0.0151 - val_loss: 0.1158 - val_acc: 0.0125
    
    Epoch 00013: val_loss improved from 0.11778 to 0.11578, saving model to 13-0.12.h5
    Epoch 14/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0163 - acc: 0.0151 - val_loss: 0.1127 - val_acc: 0.0137
    
    Epoch 00014: val_loss improved from 0.11578 to 0.11271, saving model to 14-0.11.h5
    Epoch 15/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0117 - acc: 0.0151 - val_loss: 0.1114 - val_acc: 0.0137
    
    Epoch 00015: val_loss improved from 0.11271 to 0.11141, saving model to 15-0.11.h5
    Epoch 16/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0084 - acc: 0.0151 - val_loss: 0.1200 - val_acc: 0.0137
    
    Epoch 00016: val_loss did not improve from 0.11141
    Epoch 17/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0067 - acc: 0.0151 - val_loss: 0.1110 - val_acc: 0.0125
    
    Epoch 00017: val_loss improved from 0.11141 to 0.11104, saving model to 17-0.11.h5
    Epoch 18/30
    3508/3508 [==============================] - 90s 26ms/step - loss: 0.0055 - acc: 0.0151 - val_loss: 0.1111 - val_acc: 0.0137
    
    Epoch 00018: val_loss did not improve from 0.11104
    Epoch 19/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0054 - acc: 0.0151 - val_loss: 0.1104 - val_acc: 0.0137
    
    Epoch 00019: val_loss improved from 0.11104 to 0.11038, saving model to 19-0.11.h5
    Epoch 20/30
    3508/3508 [==============================] - 92s 26ms/step - loss: 0.0049 - acc: 0.0151 - val_loss: 0.1130 - val_acc: 0.0137
    
    Epoch 00020: val_loss did not improve from 0.11038
    Epoch 21/30
    3508/3508 [==============================] - 88s 25ms/step - loss: 0.0058 - acc: 0.0151 - val_loss: 0.1138 - val_acc: 0.0137
    
    Epoch 00021: val_loss did not improve from 0.11038
    Epoch 22/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0063 - acc: 0.0151 - val_loss: 0.1144 - val_acc: 0.0125
    
    Epoch 00022: val_loss did not improve from 0.11038
    
    Epoch 00022: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 23/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0050 - acc: 0.0151 - val_loss: 0.1098 - val_acc: 0.0137
    
    Epoch 00023: val_loss improved from 0.11038 to 0.10984, saving model to 23-0.11.h5
    Epoch 24/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0038 - acc: 0.0151 - val_loss: 0.1100 - val_acc: 0.0137
    
    Epoch 00024: val_loss did not improve from 0.10984
    Epoch 25/30
    3508/3508 [==============================] - 88s 25ms/step - loss: 0.0030 - acc: 0.0151 - val_loss: 0.1099 - val_acc: 0.0137
    
    Epoch 00025: val_loss did not improve from 0.10984
    Epoch 26/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0025 - acc: 0.0151 - val_loss: 0.1103 - val_acc: 0.0137
    
    Epoch 00026: val_loss did not improve from 0.10984
    Epoch 27/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0022 - acc: 0.0151 - val_loss: 0.1102 - val_acc: 0.0137
    
    Epoch 00027: val_loss did not improve from 0.10984
    Epoch 28/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0019 - acc: 0.0151 - val_loss: 0.1107 - val_acc: 0.0137
    
    Epoch 00028: val_loss did not improve from 0.10984
    Epoch 29/30
    3508/3508 [==============================] - 89s 25ms/step - loss: 0.0017 - acc: 0.0151 - val_loss: 0.1108 - val_acc: 0.0137
    
    Epoch 00029: val_loss did not improve from 0.10984
    Epoch 30/30
    3508/3508 [==============================] - 91s 26ms/step - loss: 0.0015 - acc: 0.0151 - val_loss: 0.1111 - val_acc: 0.0137
    
    Epoch 00030: val_loss did not improve from 0.10984
    


```python
#%% 模型評估
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()  
```


```python
show_train_history(train_history, 'acc','val_acc')
show_train_history(train_history, 'loss','val_loss')
model.evaluate(x_test,y_test)
#%%
plt.scatter(y_test,model.predict(x_test))
plt.plot(y_test,y_test,'ro')
#%% 儲存模型%權重
# Save model
model.save('./faceRank.h5')
model.save_weights('./faceRank_weights.h5')
del model
from keras.models import load_model
global model
model = load_model('./faceRank.h5')
model.load_weights('./faceRank_weights.h5')
```


![png](output_10_0.png)



![png](output_10_1.png)


    1097/1097 [==============================] - 6s 5ms/step
    


![png](output_10_3.png)



```python
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

        print("Predict Score : {}".format(model.predict(img)[0][0] * 20))  
    except Exception as e :
        print(e)
        print('臉部辨識失敗')
```


```python
#%%
files = './Images'
for idx,f in enumerate(os.listdir(files)[-5:-1]):
    # 產生檔案的絕對路徑
    fullpath = os.path.join(files, f)
    predict_image(fullpath)
```

    (128, 128, 3)
    Predict Score : 41.70908451080322
    (128, 128, 3)
    Predict Score : 60.39839267730713
    (128, 128, 3)
    Predict Score : 64.27641868591309
    (128, 128, 3)
    Predict Score : 46.824235916137695
    


![png](output_12_1.png)



```python
#%%
files = './TestImages'
for idx,f in enumerate(os.listdir(files)):
    # 產生檔案的絕對路徑
    fullpath = os.path.join(files, f)
    img = load_img(fullpath)
    plt.imshow(img)
    predict_image(fullpath)
```

    (128, 128, 3)
    Predict Score : 62.73555278778076
    (128, 128, 3)
    Predict Score : 67.98409461975098
    (1,)
    Unsupported image shape: (1,)
    臉部辨識失敗
    


![png](output_13_1.png)



```python

```
