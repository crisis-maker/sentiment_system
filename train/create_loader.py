import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_get import get_list, get_data
import random


class MyDataset(Dataset):

    def __init__(self, data , label):
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.label[idx]
        return feature, label

    def __len__(self):
        return len(self.data)


def create_loader(data, label, batch_size, shuffle=True):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(20)

    dataset = MyDataset(data, label)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("the train data size is:", (train_size, data.shape[1]))
    print("the test data size is:", (test_size, data.shape[1]))

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':
    batch_size = 100
    data0 = pd.read_excel("../data_set/train/train.xlsx")
    test0 = pd.read_excel("../data_set/eval/eval.xlsx")
    # print(data.head())
    train_data, train_label = get_list(data0, 1, '文本', '情绪标签')
    train_data = get_data(train_data)
    train_label = np.array(train_label)
    test_data, test_label = get_list(test0, 1, '文本', '情绪标签')
    test_data = get_data(test_data)
    test_label = np.array(test_label)
    print("the train data size is:", train_data.shape)
    print("the test data size is:", test_data.shape)

    train_loader = create_loader(train_data, train_label, batch_size)
    for epoch in range(28):
        print("epoch:", epoch)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            print(inputs.shape)
            print(labels)
            break

