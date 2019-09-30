# k-近邻算法

"""
使用k-近邻算法进行电影分类
优点：
    精度高、对异常值不敏感、无数据输入假定
缺点：
    计算复杂度高、空间复杂度高
使用数据范围：
    数值型贺标称型
流程：
    收集数据
    准备数据
    分析数据
    训练算法
    测试算法
    使用算法
"""

import matplotlib.pyplot as plt
import numpy as np
import operator


def create_data_set():
    """
    准备数据集
    :return: 数据集以及对应标签
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['B', 'B', 'A', 'A']

    return group, labels


def classify_0(in_x, data_set, labels, k):
    """
    简单k-近邻算法实现代码
    :param in_x: 输入未知数据
    :param data_set: 训练集
    :param labels: 训练集标签
    :param k: k值
    :return: 未知数据的预测类别
    """
    date_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (date_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
            vote_i_label = labels[sorted_dist_indicies[i]]
            class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # python3 中字典的属性变为了items()
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    准备数据
    从文本文件中解析数据
    :param filename:
    :return:
    """