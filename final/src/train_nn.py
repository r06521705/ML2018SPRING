
# coding: utf-8

# In[17]:


import numpy as np
from scipy.spatial import distance
import pandas as pd
import jieba
import re
from gensim.models import word2vec
from gensim import corpora, models

from keras.models import Model,load_model
from keras.layers.core import Activation, Dense, Lambda
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Reshape,RepeatVector,TimeDistributed
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
import keras.backend.tensorflow_backend as K

import os, sys
from sys import argv
# In[18]:
training_data_file = argv[1]
testing_data_csv = argv[2]

def load_data(data_name):
    print("reading data from..." + data_name)
    x = []
    cleanr = re.compile('\n') # replace \n   
    with open(data_name,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            cleanline = re.sub(cleanr, '', line)
            x.append(cleanline)
    return x


# In[19]:


def jieba_sep_list(data):
    stopwordset
    seg_data = []
    for line in data:
        seg_line = []
        words = jieba.cut(line, cut_all=False)
        for word in words:
            if word not in stopwordset:
                seg_line.append(word)
        seg_data.append(seg_line)
    return seg_data


# In[20]:


stopwordset = set()
jieba.set_dictionary('jieba/dict_zh_tw.txt')
#stopwordset.add('的')
with open('jieba/stop_words_modified_zh_tw.txt','r',encoding='utf8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))


# In[21]:


train1 = load_data(training_data_file+'/1_train.txt')
train2 = load_data(training_data_file+'/2_train.txt')
train3 = load_data(training_data_file+'/3_train.txt')
train4 = load_data(training_data_file+'/4_train.txt')
train5 = load_data(training_data_file+'/5_train.txt')


# In[22]:


seg_train = []
seg_train.extend(train1)
seg_train.extend(train2)
seg_train.extend(train3)
seg_train.extend(train4)
seg_train.extend(train5)


# In[23]:


seg_train = jieba_sep_list(seg_train)


# In[31]:


test_data = pd.read_csv(testing_data_csv,encoding='utf8')


# In[32]:


test_data.options = test_data.options.str.replace(r'[0-9:]','')
test_data.options = test_data.options.str.replace(' ',',')
test_data.options = test_data.options.str.split('\t')
test_data.dialogue = test_data.dialogue.str.replace(' ',',')
test_data.dialogue = test_data.dialogue.str.replace('\t',',')


# In[33]:


seg_dial = jieba_sep_list(test_data.dialogue.tolist())


# In[34]:


seg_opts = []
for i in test_data.options.tolist():
    seg_opts.append(jieba_sep_list(i))


# In[24]:


from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load('models/word2vec128m1w10it50sg1.txt')


# In[25]:


model = Sequential()
model.add(Dense(128, input_shape = (128,) ))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(512, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128))
model.compile(loss="mse", optimizer='adam')

model.summary()


# In[26]:


BATCH_SIZE = 1024
w2v_dim = 128


# In[27]:


def get_batch(train_data):
    lines = []
    for line in range(len(train_data)):
        lines.append(train_data[line])
        if len(lines) == BATCH_SIZE:
            yield lines
            line -= 1
            lines = []
    yield lines
def get_w2v_train_batch(train_data):
    for lines in get_batch(train_data):

        if not lines:
            continue
        w2v_data = np.zeros((BATCH_SIZE,w2v_dim))
        for num in range(len(lines)):
            words = [word for word in lines[num] if word in word_vectors.vocab]
            if words:
                w2v_data[num] = word_vectors[words].mean(axis = 0)
        x = w2v_data[:-1]
        y = w2v_data[1:]
        yield x, y


# In[29]:


cnt = 0
for epoch in range(10):
    print('epoch '+str(epoch)+'training...')
    for batch_X, batch_Y in get_w2v_train_batch(seg_train):
        model.fit(batch_X,batch_Y,batch_size=BATCH_SIZE,verbose=0,nb_epoch=1)
        cnt += 1
        print(cnt)
    print('epoch '+str(epoch)+'done!')


# In[35]:


w2v_seg_dial = np.zeros((len(seg_dial),w2v_dim))
for num in range(len(seg_dial)):
    words = [word for word in seg_dial[num] if word in word_vectors.vocab]
    if words:
        w2v_seg_dial[num] = word_vectors[words].mean(axis = 0)


# In[36]:


ans = model.predict(w2v_seg_dial)


# In[37]:


num = 0
ans = []
for q_vec,opts in zip(w2v_seg_dial, seg_opts):
    #print(num)
    #print(opts)
    max_sim = -2
    max_sim_opt = -1
    for opt in opts:
        if opt:
            lst = [word for word in opt if word in word_vectors.vocab]
            #print(lst)
            if lst:
                opt_vector = word_vectors[lst].mean(axis=0)
                sim = 1-distance.cosine(q_vec,opt_vector)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_opt = opts.index(opt)
    if max_sim < 0.5:
        print(seg_opts.index(opts))
        num += 1
    ans.append(max_sim_opt)
    
    
print(num)


# In[40]:


with open('pred/nn_ans.csv', 'w') as f:
    f.write('id,ans\n')
    for i, v in  enumerate(ans):
        f.write('%d,%d\n' %(i, v))
print(len(ans))

