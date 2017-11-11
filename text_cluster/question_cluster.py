#encoding:utf8
import pandas as pd
import numpy as np
import jieba
import re
import pickle
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt


def prepare_train_data(df,stop_word):
    vac=[]
    train_data=[]
    for ques in tqdm(df):
        if len(ques)>10:
            doc = ""
            ques = ques.strip()
            ques = re.sub('[a-zA-Z0-9]','',ques)
            ques_cut = jieba.cut(ques)
            for word in ques_cut:
                if word not in stop_word:
                    doc += word+' '
                    if word not in vac:
                        vac.append(word)
            train_data.append(doc)
            
    train_data_path = 'train_data2.pkl'

    pickle.dump(train_data,open(train_data_path,'wb'))
    pickle.dump(vac,open('vac.pkl','wb'))

    return train_data,vac

def load_data():
    doc_set = pickle.load(open('train_data2.pkl','rb'))
    return doc_set


def k_cluster(tfidf_training_data):
    if os.path.exists('doc_cluster.pkl'):
        km = joblib.load('doc_cluster.pkl')
        clusters = km.labels_.tolist()
    else:
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        from sklearn.externals import joblib
        num_clusters = 60
     
        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()
        joblib.dump(km, 'doc_cluster.pkl')
    
#默认特征数量为30000，因为在数据预处理阶段我们发现词频数大于5个的只有30000个词左右。
def transform(dataset,n_features=30000):
    vectorizer = TfidfVectorizer(max_df=0.3, max_features=n_features, min_df= 6,use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def plot_cluster_disc(result):
    cluster_id ,cent = [],[]
    for i in range(len(result)):
        cluster_id.append(result[i][0])
        cent.append(result[i][1])
    plt.figure(figsize=(12,6))
    plt.plot(cluster_id,cent,label="cluster numbers",color="red",linewidth=1)
    plt.xlabel("cluster id")
    plt.ylabel("numbers")
    plt.legend()
    plt.show()

def train(X,vectorizer,true_k=80,minibatch = False,showLable = False,plot = False):
    #使用采样数据还是原始数据训练k-means，    
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=3000, batch_size=3000, verbose=False)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1,
                    verbose=False)
    km.fit(X)    
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print (vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :15]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    print ('Cluster distribution:')
    result_dict = dict([(i, result.count(i)) for i in result])
    result_dict = sorted(result_dict.items(),key=lambda d: d[0], reverse=False)
    print (result_dict)
    if plot:
        plot_cluster_disc(result_dict)
    return -km.score(X)

def test_k():
    '''测试选择最优聚类数量'''
    dataset = load_data()    
    print("%d documents" % len(dataset))
    X,vectorizer = transform(dataset,n_features=30000)
    true_ks = []
    scores = []
    for i in range(50,101,5):        
        score = train(X,vectorizer,true_k=i)/len(dataset)
        print (i,score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(true_ks,scores,label="error",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()
    
def test_fea():
    '''测试选择最优特征数'''
    dataset = load_data()    
    print("%d documents" % len(dataset))
    
    fea_num = []
    scores = []
    for i in range(20000,40001,2000): 
        X,vectorizer = transform(dataset,n_features=i)
        score = train(X,vectorizer,true_k=80)/len(dataset)
        print (i,score)
        fea_num.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(fea_num,scores,label="error",color="red",linewidth=1)
    plt.xlabel("features_num")
    plt.ylabel("error")
    plt.legend()
    plt.show()    

def out():
    '''在最优参数下输出聚类结果'''
    dataset = load_data()
    X,vectorizer = transform(dataset,n_features=25000)
    score = train(X,vectorizer,true_k=65,showLable=True,plot = True)/len(dataset)
    print (score)

if __name__ == "__main__": 
    # df = pd.read_csv('法律问答.csv',sep=',',header = 0,encoding='utf8')
    # df = df['question']
    # df = df.drop_duplicates()
    # df = np.array(df)
    # stop_word = pickle.load(open('stop_word_list.pkl','rb'))
    # stop_word.extend(["您好","法律","律师","谢谢"])

    # train_data,vac = prepare_train_data(df,stop_word)

    # print("词表字典大小：",len(vac))
    # print("训练数据文档数量：",len(train_data))
    # test_k()
    # test_fea()
    out()