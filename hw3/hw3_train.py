from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential , load_model
from keras.layers.core import Dense,Activation
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D  
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam
import sys
import numpy as np
import math
import csv
import pandas as pd
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF



def read_data():

    total = []
    x_train = []
    y_train = []
        
    with open(sys.argv[1], 'r') as csvfile:
        box = csv.reader(csvfile, delimiter=' ')
        for row in box:
            total.append(row)
            
    del(total[0])
    
    x_train = np.zeros((len(total),48*48),dtype = float) #先統data的資料型態以便後續的數學動作(normailze)
    y_train = np.zeros((len(total),1 ),dtype = float) 
            
    for i in range(len(total)):
        temp = np.array(total[i][0].split(",")).reshape(1,2) #np.array出來可能是string 要記得調
        temp = temp.astype(np.float)
        x_train[i][:1] = temp[0][1]
        x_train[i][1:] = total[i][1:]
        y_train[i][0] = temp[0][0]
        
        
    x_train = x_train.reshape(len(x_train),48,48,1).astype('float32') #下面的Conv2D要吃的就是4維的matrix所以要在這邊先轉成適當的大小樣式
    x_train_norm = x_train/255                                     #對灰階數字做normailization (255 ==> 0~1)
    y_train_onehot = np_utils.to_categorical(y_train, 7)

    valid_index = np.arange(len(x_train))
    np.random.shuffle(valid_index)
    index = valid_index[0:2000]
    valx, valy = x_train_norm[index], y_train_onehot[index]


    return x_train_norm , y_train_onehot , valx , valy




def build_model():

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(48,48,1)))              #根據VERYDEEPCONVOLUTIONALNETWORKS FORLARGE-SCALEIMAGERECOGNITION Karen Simonyan ∗ & Andrew Zisserman論文中
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))             #關於視覺辨識模型的建議: 1.使用多層較小的filter代替一層大的filter，能夠擁有較少參數且非線性組合更多元
        model.add(BatchNormalization())                                              #                      2.filter張數及下方一般的神經網路可設為2的n次方
        model.add(MaxPooling2D((2,2)))                                               # 可利用BatchNormalization()來加速收斂、避免overfitting、降低网络对初始化权重不敏感                      
        model.add(Dropout(0.3))                                                      

        model.add(Conv2D(64,(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.3))


        model.add(Flatten())

        model.add(Dense(units=512,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(units=7,activation='softmax'))
        model.summary() 

        return model
            



x_train,y_train, x_valid , y_valid = read_data()


datagen = ImageDataGenerator(rotation_range=10,   width_shift_range=0.1,   height_shift_range=0.1,  horizontal_flip=True)

datagen.fit(x_train)

model = build_model()


model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])


history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),  steps_per_epoch=len(x_train) / 32, epochs=120,  validation_data=[x_valid,y_valid])


model.save('hw3_model_best.h5')
score = model.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])


