import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pickle
from data_get import get_label_dict
import pandas as pd
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.rc("font",family='YouYuan')


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Oranges'))
    plt.colorbar()  # 绘制图例

    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j]+ 0.5), 'd'),
                         ha="center", va="center",
                         color="white" if cm[i][j] < thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    plt.savefig('../result/plot/' + title + '.jpg')
    plt.show()


def plot_matrix_main(path, title):
    result = pickle.load(open(path, 'rb'))
    dict_ = pickle.load(open('../result/class', 'rb'))
    # print(dict_)
    data = pd.read_excel('../dataset/train/train.xlsx')
    idx = get_label_dict(data, list(data.columns)[-1])
    y_pred = []
    y_true = []
    for i in result.keys():
        count = 0
        for j in result[i].keys():
            count += result[i][j]
            y_pred.extend([idx[j] for k in range(result[i][j])])
        y_true.extend([idx[i] for k in range(count)])
    # print(y_true,'\n',y_pred)
    axis_label = [i for i in dict_.values()]
    plot_matrix(y_true, y_pred, [i for i in range(len(result))], title=title, thresh=0.8, axis_labels=axis_label)


def plot_line(path, title):
    res = pickle.load(open(path, 'rb'))
    x = [(i+1) for i in range(len(res['loss']))]
    plt.figure(figsize=(5, 5), dpi=100)
    res['loss'] = [100 * i for i in res['loss']]
    plt.plot(x, res['loss'])
    plt.plot(x, res['acc'])
    plt.grid()
    plt.axhline(y=100, xmin=0, xmax=len(res['loss']), color='r')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Number', fontsize=14)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, len(res['loss'])+1)
    plt.ylim(0, max(res['loss'])+10)
    plt.title(title)
    plt.legend(['100 * loss', 'acc'])
    plt.savefig('../result/plot/' + title + '.jpg')
    plt.show()


def plot_comprehensive_image(paths, title):
    name = []
    plt.figure(figsize=(5, 5), dpi=100)
    plt.grid()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    for i in paths:
        res = pickle.load(open(i, 'rb'))
        name.append(i.split('/')[-1].split('.')[0])
        x = [(j+1) for j in range(len(res['loss']))]
        plt.plot(x, res['loss'])

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, len(res['loss']) + 1)
    plt.title(title)
    plt.legend(name)
    plt.savefig('../result/plot/' + title + '.jpg')
    plt.show()


if __name__ == '__main__':
    plot_matrix_main('../result/GRU.matrix', '多分类GRU模型混淆矩阵')
    plot_matrix_main('../result/RNN.matrix', '多分类RNN模型混淆矩阵')
    plot_matrix_main('../result/LSTM.matrix', '多分类LSTM模型混淆矩阵')
    plot_matrix_main('../result/GRU_.matrix', '多分类GRU(无Dropout)模型混淆矩阵')

    plot_line('../result/GRU.loss', '多分类GRU训练过程')
    plot_line('../result/LSTM.loss', '多分类LSTM训练过程')
    plot_line('../result/RNN.loss', '多分类RNN训练过程')
    plot_comprehensive_image(['../result/GRU.loss', '../result/LSTM.loss', '../result/RNN.loss'], '多分类各模型收敛速率对比')

    # plot_matrix_main('../result/GRU_b.matrix', '二分类GRU模型混淆矩阵')
    # plot_matrix_main('../result/RNN_b.matrix', '二分类RNN模型混淆矩阵')
    # plot_matrix_main('../result/LSTM_b.matrix', '二分类LSTM模型混淆矩阵')

    # plot_comprehensive_image(['../result/GRU_b.loss', '../result/LSTM_b.loss', '../result/RNN_b.loss'],
    # '二分类各模型收敛速率对比')
