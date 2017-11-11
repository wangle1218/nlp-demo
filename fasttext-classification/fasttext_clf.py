# _*_conding:utf8 _*_
import jieba
import re
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
import os


#######训练模型###########
model_path = './tmp/senti_model.model'
if os.path.exists(model_path+'.bin'):
    classifier = fasttext.load_model(model_path+'.bin')
else:
    train_file = './tmp/training.txt'
    # model_path = './experiment/fasttext-classification/senti_model.model'
    classifier = fasttext.supervised(train_file,model_path,label_prefix="__label__")

    test_file = './tmp/test.txt'
    result = classifier.test(test_file)
    print(result.precision)
    print(result.recall)

########加载停用词列表#######
stop_word_path = '../stopwords_cn.txt'
stop_word = []
with open(stop_word_path,'r') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
punction_list =list('、，。？！：；“”￥%&*@~#（）】【,.?!;:" "')
stop_word.extend(punction_list)
##################


######处理测试数据##########
def pre_data(data,stop_word):
    test_data = []
    for review in data:
        review = re.sub(r'[0-9a-zA-Z]+','',review)
        cut_review = jieba.cut(review)
        new_review = [word for word in cut_review if word not in stop_word]
        str_review = ' '.join(new_review)
        test_data.append(str_review)
    return test_data

test_text = ['这个酒店地理位置很好，交通方便，一出门就是步行街。',
            '破地方，酒店房间小，床也小，房间的灯很暗，外面噪声比较大。',
            '实惠经济，床上用品也很整洁，喜欢。',
            '工作人员素质太低了，一大早吵吵闹闹，都没睡好',
            '体验超棒']

test_data = pre_data(test_text,stop_word)

# print(test_data)

labels = classifier.predict(test_data)
print(labels)





