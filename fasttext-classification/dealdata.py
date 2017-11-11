#encoding:utf8
import re
import random


stop_word_path = './stopwords_cn.txt'  #自己准备停用词列表
stop_word = []
with open(stop_word_path,'r') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
punction_list =list('、，。？！：；“”￥%&\'*@~#（~^……）..._-】【,.?!;:" "/')
stop_word.extend(punction_list)

######处理消极评论数据#########
path = './cut_word_all/negative.txt' #数据集自己准备
training_neg = []
with open(path,'r') as f:
    for line in f.readlines():
        score_review = line.strip().split('\t')
        review = score_review[2].strip()
        review = re.sub(r'\\','',review)
        review = re.sub(r'[0-9a-zA-Z]+','',review)
        new_review = [word for word in review.split() if word not in stop_word]
        str_review = ' '.join(new_review)
        data_label = str_review+ "\t" +'__label__negative'+'\n'
        training_neg.append(data_label)

random.shuffle(training_neg)
print('training Negative reviews number:',len(training_neg))


#####处理积极评论数据#########
path = './cut_word_all/position.txt'
training_pos = []
with open(path,'r') as f:
    for line in f.readlines():
        score_review = line.strip().split('\t')
        review = score_review[2].strip()
        review = re.sub(r'\\','',review)
        review = re.sub(r'[0-9a-zA-Z]+','',review)
        new_review = [word for word in review.split() if word not in stop_word]
        str_review = ' '.join(new_review)
        data_label = str_review+'\t'+ '__label__position'+'\n'
        training_pos.append(data_label)

random.shuffle(training_pos)
print('total Position reviews number:',len(training_pos))
training_pos = training_pos[:101762]
print('Position reviews number:',len(training_pos))


######构建训练集和测试集#########
training = training_neg[:80000]+training_pos[:80000]
print('training reviews number:',len(training))

test = training_neg[80000:]+training_pos[80000:]
print('test reviews number:',len(test))

random.shuffle(training)
random.shuffle(test)

########将训练集和测试集保存######
train_out = './tmp/training.txt'
test_out = './tmp/test.txt'
with open(train_out,'a') as f:
    for data in training:
        f.write(data)

with open(test_out,'a') as f:
    for data in test:
        f.write(data)



