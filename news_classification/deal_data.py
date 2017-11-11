import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import jieba
import re
import random
import pickle

file_dir = './corpus_6_4000'

file_list = os.listdir(file_dir)

stop_word = []
with open('stopwords_cn.txt','r') as f:
    for word in f.readlines():
        stop_word.append(word.strip())
stop_word.append('\u3000')

map_list = {
            'Auto': 1, 
            'Culture': 2, 
            'Economy': 3, 
            'Medicine': 4, 
            'Military': 5, 
            'Sports': 6}

data_set = []
for file in file_list:
    doc_label = []
    file_path = file_dir + '/' + file
    with open(file_path) as f:
        data = f.read()
    data = re.sub('[a-zA-Z0-9]+','',data.strip())
    data = jieba.cut(data)
    datas = [word for word in data if word not in stop_word and len(word)>1]
    doc_label.append(datas)
    label = file.split('_')[0]
    doc_label.append(label)

    data_set.append(doc_label)

random.shuffle(data_set)
pickle.dump(data_set,open('./data/data_set.pkl','wb'))

df = pd.DataFrame(data_set,columns=['data','label'])
df['label'] = df['label'].map(map_list)

data = df['data'].tolist()
label = df['label'].tolist()

train_data = data[:16800]
test_data = data[16800:]

train_label = label[:16800]
test_label = label[16800:]

pickle.dump(train_data,open('./data/train_data.pkl','wb'))
pickle.dump(test_data,open('./data/test_data.pkl','wb'))
pickle.dump(train_label,open('./data/train_label.pkl','wb'))
pickle.dump(test_label,open('./data/test_label.pkl','wb'))













