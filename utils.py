import copy
import logging
import sys
from torch_geometric.datasets import Planetoid, Actor, WebKB
import random
import numpy as np


def data_loader(args):
    name = args.data
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid('~/桌面/data/', name, split='full')
        data = dataset[0]
    elif name in ['Actor', 'Cornell', 'Texas', 'Wisconsin']:
        if name == 'Actor':
            dataset = Actor('~/桌面/data/Actor')
        else:
            dataset = WebKB('~/桌面/data/', name)
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    else:
        raise
    args.num_node_features = dataset.num_node_features
    args.num_classes = dataset.num_classes
    return args, data


class Log(object):
    def __init__(self, args):
        self._logger = None
        self.save = args.save
        self.data = args.data
        self.time = args.time
        self.__get_logger()

    def __get_logger(self):
        if self._logger is None:
            logger = logging.getLogger("CTFGNAS")
            logger.handlers.clear()
            formatter = logging.Formatter('%(message)s')
            save_name = '{}/{}-{}.txt'.format(self.save, self.data, self.time)
            file_handler = logging.FileHandler(save_name)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            self._logger = logger
            return logger
        else:
            return self._logger

    def info(self, _str):
        self.__get_logger().info(_str)

    def warn(self, _str):
        self.__get_logger().warning(_str)


def choose_one_parent(parents):
    count = len(parents)
    index1, index2 = np.random.randint(0, count), np.random.randint(0, count)
    while index2 == index1:
        index2 = np.random.randint(0, count)
    if parents[index1].val_loss < parents[index2].val_loss:
        return index1, parents[index1]
    elif parents[index2].val_loss < parents[index1].val_loss:
        return index2, parents[index2]
    else:
        if random.random() < 0.5:
            return index1, parents[index1]
        else:
            return index2, parents[index2]


def different_random_num(a, b, x):
    r = random.randint(a, b)
    while r == x:
        r = random.randint(a, b)
    return r


def binArrayRepair(arr, status):
    arrList = []
    for i, value in enumerate(arr):
        if value == 0 or status[i] == 0:
            arrList.append(0)
        else:
            arrList.append(1)
    return arrList


def binArray2dec(arr):
    res = 0
    for i, value in enumerate(arr):
        res += value * (2 ** i)
    return res


def genoRepair(genoBin, max_len=16):
    index = max_len - 1
    while index > 0:
        if isinstance(genoBin[index - 1][0], list):
            status = True
            for j in range(index, max_len):
                if isinstance(genoBin[j][0], list):
                    if genoBin[j][0][index] == 1:
                        status = False
            if status is True:
                genoBin[index - 1] = [0, 0, 0, 0, 0]
        index -= 1
    if len(genoBin[-1]) == 5:
        genoBin[-1].pop(-1)
        genoBin[-1].pop(-1)
    genoBin_sim = copy.deepcopy(genoBin)
    index = 0
    while index < len(genoBin_sim):
        if genoBin_sim[index][0] == 0:
            index_j = index + 1
            while index_j < len(genoBin_sim):
                if isinstance(genoBin_sim[index_j][0], list):
                    genoBin_sim[index_j][0].pop(index + 1)
                index_j += 1
            genoBin_sim.pop(index)
        else:
            index += 1
    if len(genoBin_sim) == 1 and genoBin_sim[0][0] == 0:
        genoBin_sim = []
    temp = copy.deepcopy(genoBin)
    genoDec = []
    for i in range(max_len):
        if isinstance(temp[i][0], list):
            temp[i][0] = binArray2dec(temp[i][0])
    for i in range(max_len):
        genoDec += temp[i]
    return genoBin, genoBin_sim, genoDec


def dis_cluster(X, data):
    X_labels = []
    for i in range(data.y.max() + 1):
        X_label = X[data.y == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)
    dis_intra = 0.
    for i in range(data.y.max() + 1):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= data.y.max() + 1
    dis_inter = 0.
    for i in range(data.y.max() + 1 - 1):
        for j in range(i + 1, data.y.max() + 1):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(data.y.max() + 1 * (data.y.max() + 1 - 1) / 2)
    dis_inter /= num_inter
    dis_ratio = dis_inter / dis_intra
    return 1. if np.isnan(dis_ratio) else dis_ratio
