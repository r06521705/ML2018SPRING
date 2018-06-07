from keras.utils import np_utils
import numpy as np
from keras.models import Sequential , load_model
from keras.layers.core import Dense,Activation
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D ,LSTM ,Embedding
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
import _pickle as pk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import word2vec, KeyedVectors


def load_data(with_label):

        X, Y = [], []
        with open(sys.argv[1], 'r',encoding = 'utf8') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    lines = line.split(',',1)
                    X.append(lines[1])


        if(with_label):
            return(X,Y)
        else:
            return(X)




def findmax(a): 

    output = []
    for i in range(len(a)):
        if(a[i][0]>a[i][1]):
            output.append(0)
        else:
            output.append(1)
    
    with open(sys.argv[2], 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        d = 0
        for ele in output:
            writer.writerow([str(d), str(int(ele))])
            d+=1
          		





#token = pk.load(open('token_clean_sky', 'rb')) #讀入之前做好的token

#model_token = load_model('hw4_model_token_clean_sky.h5')#讀入先前利用tokenize過後的資料train的model
#model_token2 =load_model('hw4_w2v_semi.h5')
model = load_model('hw4_model_w2v_clean_sky.h5') #w2v的model


print("loadmodel")
x_test_forw2v = load_data(False)
#x_test_fortoken = load_data(False)
print("loaddata")
#del(x_test_fortoken[0])
del(x_test_forw2v[0])


mode = sys.argv[3]

word_vector = KeyedVectors.load('w2v_dimen100_clean.txt')



w2v = []
num_of_row=0


for row in x_test_forw2v:
    w2v.append([])
    for word in row.split():
        if word not in word_vector:
            w2v[num_of_row].append(word_vector['of'])
        else:
            w2v[num_of_row].append(word_vector[word])
    while len(w2v[num_of_row]) <100 :
        w2v[num_of_row].append(word_vector['of'])

    num_of_row += 1


print("w2vector")
#x_test_seq = token.texts_to_sequences(x_test_fortoken) 
#x_test_fortoken = pad_sequences(x_test_seq, maxlen=100) 
#print("tokenize testing data")

output1 = model.predict(w2v)
#output2 = model_token.predict(x_test_fortoken)
#output3 = model_token2.predict(x_test_fortoken)
findmax(output1)

