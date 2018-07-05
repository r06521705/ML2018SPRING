
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
import pandas as pd
import jieba
import re
from gensim.models import word2vec
from gensim import corpora, models
import os, sys
from sys import argv


# In[2]:

testing_data_csv = argv[1]

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


# In[3]:


stopwordset = set()
jieba.set_dictionary('jieba/dict_zh_tw.txt')
#stopwordset.add('的')
with open('jieba/stop_words_modified_zh_tw.txt','r',encoding='utf8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))


# In[4]:


#test_data = pd.read_csv('testing_data/testing_data.csv',encoding='utf8')
test_data = pd.read_csv(testing_data_csv,encoding='utf8')


# In[5]:


test_data.options = test_data.options.str.replace(r'[0-9:]','')
test_data.options = test_data.options.str.replace(' ',',')
test_data.options = test_data.options.str.split('\t')
test_data.dialogue = test_data.dialogue.str.replace(' ',',')
test_data.dialogue = test_data.dialogue.str.replace('\t',',')


# In[6]:


seg_dial = jieba_sep_list(test_data.dialogue.tolist())


# In[7]:


seg_opts = []
for i in test_data.options.tolist():
    seg_opts.append(jieba_sep_list(i))


# In[26]:


from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load('models/word2vec128m1w10it50.txt')


# In[27]:


num = 0
ans = []
for question,opts in zip(seg_dial, seg_opts):
    #print(num)
    #print(opts)
    max_sim = -2
    max_sim_opt = -1
    q_lst = [word for word in question if word in word_vectors.vocab]
    if q_lst:
        q_vec = word_vectors[q_lst].mean(axis=0)
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
    else:
        print(num)
        max_sim_opt = 0
    ans.append(max_sim_opt)
    
    num += 1
print(num)


# In[28]:


with open('pred/ans128.csv', 'w') as f:
    f.write('id,ans\n')
    for i, v in  enumerate(ans):
        f.write('%d,%d\n' %(i, v))
print(len(ans))

