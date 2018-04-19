from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential , load_model
from keras.layers.core import Dense,Activation
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D  
from keras.optimizers import  Adam
import sys
import numpy as np
import math
import csv
import pandas as pd

def load_test_data():
    
    test = []
    
    temp = pd.read_csv(sys.argv[1])
    value = temp.feature.values
    
    for ele in value:
        box = ele.split(" ")
        box = np.array(box).astype("float32")
        test.append(box)
        
    test = np.array(test)
        
    test_data = test.reshape(len(test),48,48,1).astype("float32")
    test_data_norm = test_data / 255
    
    return test_data_norm


test_data = load_test_data()

if(sys.argv[3] == "public"):
    model = load_model('hw3_model_best.h5')

if(sys.argv[3] == "private"):
    model = load_model('hw3_model_best.h5')
    

onehot = model.predict(test_data)



ans = []

for i in range(len(test_data)):
    ans.append(np.argmax(onehot[i]))
    
ans = np.array(ans).astype("float32") 

ans = ans.reshape(len(ans) , 1)    
print(ans.shape)
print(ans)   
 

with open(sys.argv[2], 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['id', 'label'])
    d = 0
    for ele in ans:
        writer.writerow([str(d), str(int(ele))])
        d+=1
    



 


