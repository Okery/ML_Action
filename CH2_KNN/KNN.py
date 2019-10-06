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
import os


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
    fr = open(filename)
    # 读取文件所有行，返回列表
    array_online = fr.readlines()
    # 获取行数
    numbers_of_lines = len(array_online)
    return_mat = np.zeros((numbers_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_online:
        # 截取所有的回车字符
        line = line.strip()
        # 按照制表符进行数据切割
        list_from_line = line.split("\t")
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    print(return_mat)

    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals =data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_vals
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set/np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_calss_test():
    hoRatio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * hoRatio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify_0(norm_mat[i, :],
                                       norm_mat[num_test_vecs:m, :],
                                       dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d"
              %(classifier_result, dating_labels[i]))
        if(classifier_result != dating_labels[i]) : error_count += 1.0
    print("the total errer rate is : %f" %(error_count/float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    from pip._vendor.distlib.compat import raw_input
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vlas = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify_0((in_arr - min_vlas)/ranges,
                                   norm_mat, dating_labels, 3)
    print("you will probably like this person:",
          result_list[classifier_result - 1])


def img2vector(filename):
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
        return return_vect


def handwriting_class_test():
    hw_labels = []
    training_file_list = os.listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % filename_str)
    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % filename_str)
        clasifier_result = classify_0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with %d, the real answer is %d"
              % (clasifier_result, class_num_str))
        if(clasifier_result != class_num_str): error_count += 1.0

    print("\n the total number of error is %d", error_count)
    print("\n the total error rate is %f", (error_count/float(m_test)))