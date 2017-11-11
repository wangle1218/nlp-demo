import numpy as np
import pickle
import os
# from gensim import models,corpora
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import classification_report  


def make_idf_vocab(train_data):
    if os.path.exists('./data/idf.pkl'):
        idf = pickle.load(open('./data/idf.pkl','rb'))
        vocab = pickle.load(open('./data/vocab.pkl','rb'))
    else:
        word_to_doc = {}
        idf = {}
        total_doc_num = float(len(train_data))

        for doc in train_data:
            for word in set(doc):
                if word not in word_to_doc.keys():
                    word_to_doc[word] = 1
                else:
                    word_to_doc[word] += 1

        for word in word_to_doc.keys():
            if word_to_doc[word] > 10:
                idf[word] = np.log(total_doc_num/(word_to_doc[word]+1))

        sort_idf = sorted(idf.items(),key=lambda x:x[1])
        vocab = [x[0] for x in sort_idf]
        pickle.dump(idf,open('./data/idf.pkl','wb'))
        pickle.dump(vocab,open('./data/vocab.pkl','wb'))
    return idf,vocab

def cal_term_freq(doc):
    term_freq = {}
    for word in doc:
        if word not in term_freq.keys():
            term_freq[word] = 1
        else:
            term_freq[word] += 1
    for word in term_freq.keys():
        term_freq[word] = term_freq[word]/float(len(doc))
    return term_freq

def make_doc_feature(vocab,idf,doc,topN):
    doc_feature = [0.]*topN
    vocab = vocab[:topN]
    tf = cal_term_freq(doc)
    for word in doc:
        if word in vocab:
            index = vocab.index(word)
            doc_feature[index] = tf[word]*idf[word]
    return doc_feature

def make_tfidf(train_data,vocab,idf,topN):
    tfidf_data = []
    for doc in train_data:
        doc_feature = make_doc_feature(vocab,idf,doc,topN)
        tfidf_data.append(doc_feature)
    return tfidf_data

if __name__ == '__main__':
    train_data = pickle.load(open('./data/train_data.pkl','rb'))
    train_label = pickle.load(open('./data/train_label.pkl','rb'))

    idf,vocab = make_idf_vocab(train_data)
    tfidf_data = make_tfidf(train_data,vocab,idf,6000)
    train_x = np.array(tfidf_data[:13500])
    train_y = np.array(train_label[:13500])
    val_x = np.array(tfidf_data[13500:])
    val_y = np.array(train_label[13500:])

    clf = MultinomialNB().fit(train_x, train_y)
    predicted = clf.predict(val_x) 
    acc = np.mean(predicted == val_y)
    print(' validation accuracy rate:',acc)  

    test_data = pickle.load(open('./data/test_data.pkl','rb'))
    test_label = pickle.load(open('./data/test_label.pkl','rb'))
    tfidf_test = make_tfidf(test_data,vocab,idf,6000)
    tfidf_test = np.array(tfidf_test)

    predicted = clf.predict(tfidf_test)
    acc = np.mean(predicted == test_label)
    print('test accuracy rate:',acc)

    









