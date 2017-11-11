#encoding=utf8

import re
import collections
from operator import mul
from functools import reduce
import pickle
import os
import math

def load_corpus(file_path):
    corpus = {}
    # corpus_scale = 0
    with open(file_path,'r') as f:
        for line in f.readlines():
            line_words = line.strip().split()
            if len(line_words) > 1:
                for word in line_words:
                    word = re.findall('[a-z]+', word.lower())
                    if len(word) > 0:
                        word = '>'+word[0]+'<'
                        if word in corpus.keys():
                            corpus[word] += 1
                        else:
                            corpus[word] = 1

    pickle.dump(corpus,open('corpus.pkl','wb'))
    return corpus

def frequency(symbol,corpus):
    l = len(symbol)
    freq = 0
    for word in corpus.keys():
        freq_i = 0
        for i in range(len(word)):
            if l == 1:
                if word[i] == symbol:
                    freq_i += 1
            if l == 2:
                if word[i:i+2] == symbol:
                    # print(word)
                    freq_i += 1
        freq_i = freq_i * corpus[word]
        freq += freq_i
    return freq 

def condition_prob(w1,w2,corpus):
    freq_w1 = frequency(w1,corpus)
    freq_w2 = frequency(w2,corpus)
    return (1+float(freq_w2))/(len(corpus.keys())+float(freq_w1))

def testing(word,corpus):
    cond_probs = []
    cond_p = condition_prob('>','>'+word[0],corpus)
    cond_probs.append(cond_p)

    for i in range(len(word)-1):
        cond_p = condition_prob(word[i],word[i:i+2],corpus)
        cond_probs.append(cond_p)

    cond_p = condition_prob(word[-1],word[-1]+'<',corpus)
    cond_probs.append(cond_p)
 
    reliability = reduce(mul, cond_probs) * math.pow(10,len(word))
    return reliability



if __name__ == "__main__":
    if os.path.exists('corpus.pkl'):
        corpus = pickle.load(open('corpus.pkl','rb'))
    else:
        corpus = load_corpus('corpus.txt')

    test = "ni want yao zhidao how many yingyu words you have learned xuexi tiantian xiangshang"
    test = test.split()
    for test_word in test:
        test_word = test_word.lower()
        # print(test_word)
        reliability = testing(test_word,corpus)
        # print(reliability)
        if reliability >= 1e-3:
            print("%s  not a pingyin" % test_word)
        else:
            print("%s is a pingyin" % test_word)



