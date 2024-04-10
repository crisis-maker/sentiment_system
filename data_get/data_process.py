import re
import numpy as np
import pandas as pd
import pickle
import jieba
from collections import Counter


def get_label_dict(data, X):
    lab = {}
    idx = 0
    for i in range(data.shape[0]):
        if data.loc[i, X] not in lab:
            lab[data.loc[i, X]] = idx
            idx += 1
    return lab


def get_list(data, flag=0, X=None, y=None):
    if flag:
        a_data = data[X]
        # print(a_data.head())
        a_data = a_data.apply(
            lambda x: str(x).replace(' ', "").replace(r'\t', "").replace('//', "").replace('[', "").replace(']', ""))
        a_data = a_data.apply(lambda x: re.sub('#', '', str(x)))
        a_data = a_data.apply(lambda x: re.sub('http*', '', str(x)))
        a_data = a_data.apply(lambda x: re.sub('\d', '', str(x)))
        for i in range(a_data.shape[0]):
            if '@' in a_data[i] and ':' in a_data[i]:
                a_data.loc[i] = a_data.loc[i][:a_data.loc[i].index('@')]+a_data.loc[i][a_data.loc[i].index(':')+1:]

        data2 = pd.DataFrame([a_data, data[y]], index=['文本', '标签']).T
        # data2.to_excel('../dataset/train/b_train.xlsx')
        ans_data = []
        ans_label = []
        lab_dict = get_label_dict(data, y)
        for i in range(data2.shape[0]):
            ans_data.append(data2.loc[i, '文本'])
            a = [0 for i in range(len(lab_dict))]
            a[lab_dict[data2.loc[i, '标签']] - 1] = 1
            ans_label.append(a)
        return ans_data, ans_label
    else:
        ans_data = []
        for line in data:
            line = line.strip('\n')
            if len(line) > 0:
                ans_data.extend(line.split('。'))
        return ans_data


def cut_word(data):
    return [jieba.lcut(i) for i in data]


def max_len(data):
    max_ = 0
    for i in data:
        if len(i) > 200:
            # print(data.index(i))
            max_ = 200
            break
        if len(i) > max_:
            max_ = len(i)

    print('The max len is：', max_)
    return max_


def pad_input(sentences, seq_len):
    """
    将句子长度固定为`seq_len`，超出长度的从后面截断，长度不足的在前面补0
    """
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def get_data(data, test_flag=0, word2idx=None):
    if test_flag==0:
        data = cut_word(data)
        words = Counter()  # 用于统计每个单词出现的次数
        for i, words_list in enumerate(data):
            words.update(words_list)  # 更新词频列表

        print("cut and count done")
        words = {k: v for k, v in words.items() if v > 1}
        word2idx = {o: i for i, o in enumerate(words)}
        idx2word = {i: o for i, o in enumerate(words)}
        for i, sentence in enumerate(data):
            data[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
        m = max_len(data)
        sentences = pad_input(data, m)
        word2idx['len'] = len(words)
        word2idx['max_len'] = m
        print("feature extract done")

        return sentences, len(words), word2idx
    else:
        data = cut_word(data)
        for i, sentence in enumerate(data):
            data[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
        sentences = pad_input(data, int(word2idx['max_len']))
        print("feature extract done")

        return sentences


if __name__ == '__main__':
    data0 = pd.read_csv("../dataset/train/weibo_senti_100k.csv")
    # test0 = pd.read_excel("../dataset/eval/eval.xlsx")
    # print(data.head())
    train_data, train_label = get_list(data0, 1, 'review', 'label')
    train_data, len_, pro_dict = get_data(train_data)
    pickle.dump(pro_dict, open('./dict_', "wb"), protocol=4)
    # print(data[:10])
    # print(data_[:10],label[:10])
    # with open("C:\\Users\\llhy\\Desktop\\竞赛\\蓝桥\\practice\\并查集\\排座位.txt",encoding='utf-8') as f:
    #     print(f.readlines())
