from data_get import get_label_dict
import pandas as pd
import pickle


def senti_dict(path):
    data = pd.read_excel(path)
    dict_ = get_label_dict(data, list(data.columns)[-1])
    dict_idx = {}
    idx = 0
    for i in dict_.keys():
        dict_idx[idx] = i
        idx += 1

    return dict_idx


def count_num(path):
    data = pd.read_excel(path)
    dict_num = {}
    for i in range(data.shape[0]):
        if data.loc[i, list(data.columns)[-1]] not in dict_num:
            dict_num[data.loc[i, list(data.columns)[-1]]] = 1
        else:
            dict_num[data.loc[i, list(data.columns)[-1]]] += 1

    return dict_num


if __name__ == '__main__':
    pickle.dump(senti_dict("../dataset/train/b_train.xlsx"), open('../result/class_', "wb"), protocol=4)
    df = pd.DataFrame(count_num('../dataset/train/b_train.xlsx').items(), columns=['emotion', 'amounts'])
    df.to_excel('../result/b_data_num.xlsx')
