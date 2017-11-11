import numpy as np
import jieba
import re
import pickle
from nltk.probability import  FreqDist,ConditionalFreqDist 
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from random import shuffle
import sklearn
from nltk.classify.scikitlearn import  SklearnClassifier
from sklearn.svm import SVC, LinearSVC,  NuSVC
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score

pos_list = []
neg_list = []
stop_word_list = []

pos_cut_list = []
neg_cut_list = []

pos_path = './data/pos.txt'
neg_path = './data/neg.txt'
data_cut_path = './data/pos_cut.pkl'
stop_word = './data/stoplist.txt'

with open(stop_word,'r') as fs:
    for line in fs.readlines():
        stop_word_list.append(line.strip())                               #停用词列表

#数据清洗和切分
def cut_data(data_path,data_cut_path):
    with open(data_path,'r') as fp:
        for line in fp.readlines():
            line = re.sub(r'[a-zA-Z0-9]','',line)                          #清楚评论数据中的数字和字母
            pos_list.append(line)
        
    pos_list = np.array(pos_list)
    pos_list = np.unique(pos_list)                                         #评论数据集去重

    for comment in pos_list:
        comment_cut = jieba.cut(comment)                                   #将每一条评论进行分词
        comment_cut_str = []
        for word in comment_cut:
            if word not in stop_word_list:
                comment_cut.append(word)

        pos_cut_list.append(comment_cut)
                
    print('len(pos_cut_list):',len(pos_cut_list))

    pickle.dump(pos_cut_list,open(data_cut_path,'wb'))

#把单个词作为特征
def bag_of_words(words):
    return dict([(word,True) for word in words])

#把双个词作为特征--使用卡方统计的方法，选择排名前1000的双词
def bigram(words,n=1000):
    score_fn = BigramAssocMeasures.chi_sq
    bigram_finder = BigramCollocationFinder.from_words(words)      #把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn,n)                    #使用卡方统计的方法，选择排名前1000的双词 
    newBigrams = [u+v for (u,v) in bigrams]
    return bag_of_words(newBigrams)

#把单个词和双个词一起作为特征
def bigram_words(words,n=1000):  
    score_fn = BigramAssocMeasures.chi_sq
    bigram_finder = BigramCollocationFinder.from_words(words)  
    bigrams = bigram_finder.nbest(score_fn,n) 
    newBigrams = [u+v for (u,v) in bigrams]  
    a = bag_of_words(words)
    b = bag_of_words(newBigrams) 
    a.update(b)                                                 #把字典b合并到字典a中 
    return a                                                    #所有单个词和双个词一起作为特征

#获取信息量较高(前number个)的特征(卡方统计) 
def jieba_feature(number):     
    pos_words = []  
    neg_words = [] 
    for items in pickle.load(open('./data/pos_cut.pkl','rb')):                     #把集合的集合变成集合 
        for item in items:
            pos_words.append(item)
    for items in pickle.load(open('./data/neg_cut.pkl','rb')):
        for item in items:
            neg_words.append(item)
  
    word_fd = FreqDist()                                       #可统计所有词的词频
  
    cond_word_fd = ConditionalFreqDist()                       #可统计积极文本中的词频和消极文本中的词频
  
    for word in pos_words:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
  
    for word in neg_words:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1
  
    pos_word_count = cond_word_fd['pos'].N()                    #积极词的数量
  
    neg_word_count = cond_word_fd['neg'].N()                    #消极词的数量
  
    total_word_count = pos_word_count + neg_word_count
  
    word_scores = {}                                            #包括了每个词和这个词的信息量
  
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word],  (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],  (freq, neg_word_count), total_word_count) #同理

        word_scores[word] = pos_score + neg_score               #一个词的信息量等于积极卡方统计量加上消极卡方统计量
  
    best_vals = sorted(word_scores.items(), key=lambda item:item[1],  reverse=True)[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
  
    best_words = set([w for w,s in best_vals])
  
    return dict([(word, True) for word in best_words])
  
    
#构建训练需要的数据格式：
  
#[[{'买': 'True', '京东': 'True', '物流': 'True', '包装': 'True', '\n': 'True', '很快': 'True', '不错': 'True', '酒': 'True', '正品': 'True', '感觉': 'True'},  'pos'],
  
# [{'买': 'True', '\n':  'True', '葡萄酒': 'True', '活动': 'True', '澳洲': 'True'}, 'pos'],
  
# [{'\n': 'True', '价格': 'True'}, 'pos']]
  
def build_features(dimension = 300):
    #四种特征选取方式，越来越好
  
    #feature = bag_of_words(text())#单个词  
    #feature = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=dimension)#双个词
    #feature =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=dimension)#单个词和双个词
  
    feature = jieba_feature(dimension)                            #结巴分词
    posFeatures = []
  
    for items in pickle.load(open('./data/pos_cut.pkl','rb')):
        a = {}
        for item in items:
            if item in feature.keys() and item != ' ':
                a[item]='True'
  
        posWords = [a,'pos']                                        #为积极文本赋予"pos"
        posFeatures.append(posWords)
  
    negFeatures = []
  
    for items in pickle.load(open('./data/neg_cut.pkl','rb')):
        a = {}
        for item in items:
            if item in feature.keys() and item != ' ':
                a[item]='True'
  
        negWords = [a,'neg']                                       #为消极文本赋予"neg"
        negFeatures.append(negWords)
  
    return posFeatures,negFeatures
  
posFeatures,negFeatures =  build_features()                       #获得训练数据
print(len(posFeatures),len(negFeatures))

shuffle(posFeatures)                                             #把文本的排列随机化  
shuffle(negFeatures)                                             #把文本的排列随机化

train =  posFeatures[1500:5500]+negFeatures[1500:] 
test = posFeatures[:1500]+negFeatures[:1500]      
shuffle(train)
shuffle(test)

# print(train[:5])
# print(test[:5])
  
data,tag = zip(*test)                                            #分离测试集合的数据和标签，便于验证和测试
# print(data[5])
def score(classifier):
    classifier = SklearnClassifier(classifier)                   #在nltk中使用scikit-learn的接口
    classifier.train(train)                                      #训练分类器

    pred = classifier.classify_many(data)                        #对测试集的数据进行分类，给出预测的标签
    n = 0
    s = len(pred)
    for i in range(0,s):
        if pred[i] == tag[i]:
            n = n+1
  
    return n/s

#设置不同维度特征，寻找最佳的特征数量和最佳的分类器
for dim in [200,500,1000,1500,2000,2500,3000]:
    posFeatures,negFeatures =  build_features(dim)
    shuffle(posFeatures)  
    shuffle(negFeatures) 

    train =  posFeatures[1500:5500]+negFeatures[1500:] 
    test = posFeatures[:1500]+negFeatures[:1500]  
    shuffle(train)
    shuffle(test)

    data,tag = zip(*test)  
    
    print('---------------------\ndimension = %d\n' %dim)

    print('BernoulliNB`s accuracy is %f'  %score(BernoulliNB()))  
    print('MultinomiaNB`s accuracy is %f'  %score(MultinomialNB()))
    print('LogisticRegression`s accuracy is  %f' %score(LogisticRegression()))
    print('SVC`s accuracy is %f'  %score(SVC()))
    print('LinearSVC`s accuracy is %f'  %score(LinearSVC()))
    print('NuSVC`s accuracy is %f'  %score(NuSVC()))