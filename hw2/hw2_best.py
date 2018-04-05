import os, sys
import csv
from sys import argv
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

from keras.models import Sequential
from keras import regularizers
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy

def load_data(train_csv_path, test_csv_path, train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=None)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
   
    #讀train.csv
    train_csv = pd.read_csv(train_csv_path, sep=',', header=0)
    train_csv = np.array(train_csv.values)
    train_csv = train_csv.T
    edu_num_train = train_csv[4].T
    #讀test.csv
    test_csv = pd.read_csv(test_csv_path, sep=',', header=0)
    test_csv = np.array(test_csv.values)
    test_csv = test_csv.T
    edu_num_test = test_csv[4].T
    
    #把train.csv的edu_num併到X_Train
    X_train_T = X_train.T
    edu_num_train = edu_num_train.reshape(1,32561)
    a = np.concatenate((X_train_T, edu_num_train))
    a = a.T
    #把test.csv的edu_num併到X_Test
    X_test_T = X_test.T
    edu_num_test = edu_num_test.reshape(1,16281)
    b = np.concatenate((X_test_T, edu_num_test))
    b = b.T
   
    discarded_columns = [7,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,54,65,66,67,68,69,70,71,72,73,74,75,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,121,122]
    #discarded_columns = [0,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,54,71,72,73,74,75,76,77,116]
    #81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,121,122,116]
    discarded_X_train = np.delete(a, discarded_columns, 1)
    discarded_X_test = np.delete(b, discarded_columns, 1)
    discarded_X_train = discarded_X_train.astype(int)
    discarded_X_test = discarded_X_test.astype(int)
    return (discarded_X_train, Y_train, discarded_X_test)
   
#sigma normalize
def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

X_train, Y_train, X_test = load_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
X_train, X_test = normalize(X_train, X_test)


#----


model = Sequential()
#model.add(Dense(38,input_dim=38,activation='sigmoid',use_bias=True,kernel_initializer='Zeros',bias_initializer='zeros') )
model.add(Dense(38,input_dim=38,activation='sigmoid' ))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.001),loss=binary_crossentropy,metrics=[binary_accuracy])
#model.fit(X_train,Y_train,batch_size=32,epochs=50,verbose=2,validation_split=0.1)
model.fit(X_train,Y_train,batch_size=32,epochs=15,verbose=2)


predict_y = model.predict(X_test)
predict_y_squeezed = np.squeeze(predict_y)


rounded_y = []
for i in range(len(predict_y_squeezed)):
	if predict_y_squeezed[i] >= 0.5:
		rounded_y.append(1)
	else:
		rounded_y.append(0)


output = open(sys.argv[6], "w+")
s=csv.writer(output, delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(rounded_y)):
	s.writerow([i+1,rounded_y[i]] )
output.close()
print("output csv file has been created")



