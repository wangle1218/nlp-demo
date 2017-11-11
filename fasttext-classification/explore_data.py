#encoding:utf8
import matplotlib.pyplot as plt

neg_path = './cut_word_all/negative.txt'
pos_path = './cut_word_all/position.txt'

def explore(path):
    sent_len = {}
    with open(path,'r') as f:
        for line in f.readlines():
            line_len = len(line)
            if line_len in sent_len.keys():
                sent_len[line_len] += 1
            else:
                sent_len[line_len] = 1

    sort_sent_len = sorted(sent_len.items(),key = lambda d:d[0])
    len_list = [x[0] for x in sort_sent_len]
    fre_list = [x[1] for x in sort_sent_len]
    return len_list,fre_list

neg_len_list,neg_fre_list = explore(neg_path)
pos_len_list,pos_fre_list = explore(pos_path)

plt.figure(1)
plt.subplot(211)
plt.scatter(neg_len_list,neg_fre_list,color="blue")
#设置x轴范围
plt.xlim(-1, 100)
plt.title('review\'s length statistic')
# plt.xlabel('review length')
plt.ylabel('frequency')

plt.subplot(212)
plt.scatter(pos_len_list,pos_fre_list,color="red")
plt.xlim(-1, 100)
plt.xlabel('review length')
plt.ylabel('frequency')
plt.show()




