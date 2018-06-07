from keras.utils import np_utils
import numpy as np
from keras.models import Sequential , load_model
from keras.layers.core import Dense,Activation
from keras.layers import Dropout,Flatten,Conv1D,MaxPooling1D ,LSTM ,Embedding,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import _pickle as pk
import sys
import numpy as np
import math
import csv
import pandas as pd
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from gensim.models import word2vec, KeyedVectors


def load_data():
    X=[]
    Y=[]
    with open(sys.argv[1], 'r' , encoding = 'utf-8') as temp:
        for line in temp:
            handle = line.strip().split(' +++$+++ ')
            X.append(handle[1])
            Y.append(int(handle[0]))    
    Y = np_utils.to_categorical(Y, 2)     #one-hot encoding      

    return X,Y
    
def word2vector(x_train):
    w2v = []
    numofrow = 0
    word_vector = KeyedVectors.load('w2v_dimen100_clean.txt') #interface　讀取 將word轉成vector
    for row in x_train:
        w2v.append([])
        for unit in row.split():
            if unit not in word_vector:
                w2v[numofrow].append(word_vector['of'])      #上一步將每個單詞轉成一個100維的vector,這裡又因為後面的model一開始是吃100維的資料
            else:                                            #所以w2v的第一筆等等要input到model裡的資料是 100 個 100維的資料(100*100)
                w2v[numofrow].append(word_vector[unit])      #這裡因為在做interface時min count是設50 亦即 在訓練interface時資料中字詞出現少於50次者
        while len(w2v[numofrow]) < 100 :                     #不考慮進來，所以不去計算其詞向量，所以在這邊要對廢字給予一個自己訂的向量(ex: of)
            w2v[numofrow].append(word_vector['of'])          #因為of這種介係詞不太會影響語意，相當於一個無作用的向量
        numofrow = numofrow + 1  
    return w2v                                               #這裡會說不足100補到100(選擇100這個數字)可能是因為training data中最長的詞句可能又是100
                        

def build_model_w2v():

        model = Sequential()
        model.add(LSTM(128,dropout = 0.2,recurrent_dropout = 0.2,input_shape = (100,100)))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='sigmoid'))
        model.summary() 

        return model            
        

            
nolabel = sys.argv[2]
x_train , x_label = load_data()    # 小心x_label已經處理過變成one_hot-encoding 2維(0,1)或(1,0)
print("load_data")
w2v = word2vector(x_train)
print("transtow2v")
model = build_model_w2v()
print("bulidmodel")

earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max') #當準確率不再提升或再訓練下去或降低model表現時終止訓練
                                                                                      # monitor 跟mode互相搭配              
                                                                                      #patience ==> 可以忍受幾次結果不變                      
checkpoint = ModelCheckpoint(filepath='hw4_model_w2v_clean_sky.h5',    #在訓練過程中將最佳的結果儲存出來
                                verbose=1,
                                save_best_only=True,
                                monitor='val_acc',
                                mode='max' )



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(w2v,x_label,batch_size=128,epochs=6,validation_split=0.08,shuffle=True,callbacks=[checkpoint, earlystopping])


score = model.evaluate(w2v,x_label)
print ('\nTrain Acc:', score[1])


#EarlyStopping是Callbacks的一种，callbacks用于指定在每个epoch开始和结束的时候进行哪种特定操作。
#Callbacks中有一些设置好的接口，可以直接使用，如’acc’,’val_acc’,’loss’和’val_loss’等等。 
#EarlyStopping则是用于提前停止训练的callbacks。具体地，可以达到当训练集上的loss不在减小（即减小的程度小于某个阈值）的时候停止继续训练。
