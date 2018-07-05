
# coding: utf-8

# In[34]:


import numpy as np
from scipy.spatial import distance
import pandas as pd
import jieba
import re
from gensim.models import word2vec
from gensim import corpora, models
import os, sys
from sys import argv

# In[35]:

training_data_file = argv[1]

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


# In[36]:


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


# In[37]:


stopwordset = set()
jieba.set_dictionary('jieba/dict_zh_tw.txt')
#stopwordset.add('的')
with open('jieba/stop_words_modified_zh_tw.txt','r',encoding='utf8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))


# # training data processing

# In[38]:


train1 = load_data(training_data_file+'/1_train.txt')
train2 = load_data(training_data_file+'/2_train.txt')
train3 = load_data(training_data_file+'/3_train.txt')
train4 = load_data(training_data_file+'/4_train.txt')
train5 = load_data(training_data_file+'/5_train.txt')


# In[39]:


seg_train = []
seg_train.extend(train1)
seg_train.extend(train2)
seg_train.extend(train3)
seg_train.extend(train4)
seg_train.extend(train5)


# In[40]:


seg_train = jieba_sep_list(seg_train)


# # save seg file and load it as Text8Corpus

# In[41]:


with open('seg/allseg.txt','w',encoding='utf8') as output:
    for line in seg_train:
        for word in line:
            output.write(word+' ')
        output.write('\n')


# In[42]:


sentences = word2vec.Text8Corpus('seg/allseg.txt')


# # train w2v model

# In[43]:


dim = 64
min_count = 1
window = 20
iteration = 150
sg = 1
neg = 5
note = 'all'
fname = str(dim)+'m'+str(min_count)+'w'+str(window)+'it'+str(iteration)+'sg'+str(1)+'neg'+str(neg)+note


# In[44]:


model = word2vec.Word2Vec(sentences, size=dim, min_count = min_count, window=window , iter = iteration, sg=sg, negative=neg)
model.wv.save("models/word2vec"+fname+".txt") #modelPath
word_vectors = model.wv
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load("models/word2vec"+fname+".txt")


print(training_data_file)