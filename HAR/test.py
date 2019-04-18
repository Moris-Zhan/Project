#%%
import pandas as pd
import copy
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#%% 1.	Overview
orig_train_df = pd.read_csv('pml-training.csv')
orig_train_df.head()
orig_train_df.describe()
orig_train_df.count()

orig_val_df = pd.read_csv('pml-testing.csv')
orig_val_df.head()
orig_val_df.describe()
orig_val_df.count()
#%%
[print(c) for c in orig_train_df.columns] # origin columns
[print(c) for c in orig_val_df.columns] # origin columns

#%% 2.	Background
# Unnamed: 0 : index-->不重要
# user_name : 使用者名字
# raw_timestamp_part_1 : 時間戳記1
# raw_timestamp_part_2 : 時間戳記2
# cvtd_timestamp : 測試日期
# new_window (yes/no) : 測試窗口 -->不重要
# num_window  : 測試窗口數-->不重要
# classe: 預測類別(Label) - training
# problem_id: 錯誤代碼 - testing

# 皮帶測量值(一般/陀螺儀/加速度/強度/3方向) 
# roll_belt
# pitch_belt
# yaw_belt
# total_accel_belt
# gyros_belt_x
# gyros_belt_y
# gyros_belt_z
# accel_belt_x
# accel_belt_y
# accel_belt_z
# magnet_belt_x
# magnet_belt_y
# magnet_belt_z

# 手腕測量值(陀螺儀/加速度/強度/3方向)
# gyros_arm_x
# gyros_arm_y
# gyros_arm_z
# accel_arm_x
# accel_arm_y
# accel_arm_z
# magnet_arm_x
# magnet_arm_y
# magnet_arm_z

# 啞鈴測量值(一般/總體/陀螺儀/加速度/強度/3方向)
# roll_dumbbell
# pitch_dumbbell
# yaw_dumbbell
# total_accel_dumbbell
# gyros_dumbbell_x
# gyros_dumbbell_y
# gyros_dumbbell_z
# accel_dumbbell_x
# accel_dumbbell_y
# accel_dumbbell_z
# magnet_dumbbell_x
# magnet_dumbbell_y
# magnet_dumbbell_z

# 前臂測量值(一般/總體/陀螺儀/加速度/強度/3方向)
# roll_forearm
# pitch_forearm
# yaw_forearm
# total_accel_forearm
# gyros_forearm_x
# gyros_forearm_y
# gyros_forearm_z
# accel_forearm_x
# accel_forearm_y
# accel_forearm_z
# magnet_forearm_x
# magnet_forearm_y
# magnet_forearm_z

# 皮帶測量值(波峰/偏移/最大/最小/震幅/方差/平均/標準差) 
# 取決於 new_window
# kurtosis_roll_belt 
# kurtosis_picth_belt
# kurtosis_yaw_belt
# skewness_roll_belt
# skewness_roll_belt
# skewness_yaw_belt
# max_roll_belt
# max_picth_belt
# max_yaw_belt
# min_roll_belt
# min_pitch_belt
# min_yaw_belt
# amplitude_roll_belt
# amplitude_pitch_belt
# amplitude_yaw_belt
# var_total_accel_belt
# var_roll_belt
# var_yaw_belt
# var_pitch_belt
# avg_roll_belt
# avg_pitch_belt
# avg_yaw_belt
# stddev_roll_belt
# stddev_pitch_belt
# stddev_yaw_belt


#%% 3.	Data Exploration
def Clean(df):
    df = df.fillna(0)
    # 挑選DROP欄位
    dropCol = []
    # key = '#DIV/0!'
    key = 0
    for col in df.columns:
        count = list(df[col]).count(key)
        if count > df[col].count() / 2:
            dropCol.append(col)
        # print("Column:{} , Containin {} 個{}".format(col,count,key))
    # print("Column Drop:{}".format(dropCol))
    # print()
    for col in dropCol:
        df = df.drop(col,axis=1)  
    
    dropCol=['Unnamed: 0','new_window','num_window','problem_id']
    for col in dropCol:
        if col in df.columns:
            df = df.drop(col,axis=1)  

    return df

def Argu(df):
    def ts(row):
        return str(row['raw_timestamp_part_1'])+ "."+ str(row['raw_timestamp_part_2'])
    df["cvtd_timestamp"] = pd.to_datetime(df["cvtd_timestamp"])
    df["raw_timestamp"] = df.apply(ts,axis=1)
    df["raw_timestamp"] = pd.to_datetime(df["raw_timestamp"], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df["raw_timestamp"] = pd.to_datetime(df["raw_timestamp"])
    df = df.drop('raw_timestamp_part_1',axis=1)  
    df = df.drop('raw_timestamp_part_2',axis=1)     
    
    def sub_dateTime(row):
        sub = (row['raw_timestamp'] - row['cvtd_timestamp']).days
        return sub

    df['Interval'] = df.apply(sub_dateTime,axis=1)
    # df[''] = df['data1'].groupby(df['key1'])
    df = df.sort_values(by=['user_name','cvtd_timestamp','raw_timestamp'])     
    df = df.reset_index(drop=True)
    df = df.drop('cvtd_timestamp',axis=1)  
    df = df.drop('raw_timestamp',axis=1)  
    df = df.drop('user_name',axis=1) 

    if 'classe' in df.columns:
        # df['classe'] = df['classe'].map(cat_mapping)
        global class_le
        class_le = LabelEncoder()
        df['classe'] = class_le.fit_transform(df['classe'].values)

    # if 'classe' in df.columns:
    #     np_utils.to_categorical(df['classe'], num_classes=5)

    return df

train_df = pd.read_csv('pml-training.csv')
train_df = Clean(train_df)
train_df = Argu(train_df)

val_df = pd.read_csv('pml-testing.csv')
val_df = Clean(val_df)
val_df = Argu(val_df)

train_df.head()
# [c for c in train_df.columns] # remain columns
# [c for c in val_df.columns] # remain columns


#%% 4.	Prediction Modeling
feaCols = [col  for col in train_df.columns if col!="classe"]
print(feaCols)

X_train,X_test,y_train,y_test = train_test_split(
    train_df[feaCols],
    train_df["classe"],
    test_size = 0.2,
    random_state = 0
)

sc = StandardScaler()
sc.fit(X_train)
X_train_Std = sc.transform(X_train)
X_test_Std = sc.transform(X_test)

y_TrainOneHot = np_utils.to_categorical(y_train,num_classes=5)
y_TestOneHot  = np_utils.to_categorical(y_test,num_classes=5)

#%%

def buildModel(shape):
    model = Sequential()
    # Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
    model.add(Dense(units=256, input_dim=shape[1], kernel_initializer='normal', activation='relu')) 
    # Add output layer
    model.add(Dense(units=5, kernel_initializer='normal', activation='softmax'))
    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

    return model

print(X_train_Std.shape)
model = buildModel(X_train_Std.shape)
# # 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=X_train_Std, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=32, verbose=2)  

# # 顯示訓練成果(分數)
scores = model.evaluate(X_test_Std, y_TestOneHot)  
print('Loss :{} , Accuracy:{} '.format(scores[0],scores[1]))



#%% 5.	Model Application
sc = StandardScaler()
sc.fit(val_df)
X_val_Std = sc.transform(val_df)

val_df['classe'] = model.predict_classes(X_val_Std)
orig_val_df['classe'] = class_le.inverse_transform(val_df['classe'].values)
#%%
orig_val_df.head()

#%%
