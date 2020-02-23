# -*-coding2222:utf8-*-
"""
author:zhiyuan
date:20190316
test the performance of the model in the test data for gbdt and gbdt_lr
"""
from __future__ import division #防止整数的除法结果转换为整数，而是变成精确的浮点数
import numpy as np
import xgboost as xgb
import production.train as TA
from scipy.sparse import csc_matrix
import math
import sys


def get_test_data(test_file, feature_num_file):
    """
    Args:
        test_file:file to check performance
        feature_num_file: the file record total num of feature
    Return:
         two np array: test _feature, test_label
    """
    total_feature_num = 103
    test_label = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= feature_list)
    return test_feature, test_label


# 打分函数
def predict_by_tree(test_feature, tree_model):
    """
    predict by gbdt model
    """
    predict_list = tree_model.predict(xgb.DMatrix(test_feature)) #调用GBDT模型预测函数
    return predict_list

#GBDT和LR混合模型打分函数
def predict_by_lr_gbdt(test_feature, mix_tree_model, mix_lr_coef, tree_info):
    """
    mix_lr_coef: LR模型参数列表（和GBDT模型输出的特征维数相同）
    predict by mix model
    """
    tree_leaf = mix_tree_model.predict(xgb.DMatrix(test_feature), pred_leaf = True) #得到每个样本在GBDT模型中落在哪个节点上
    (tree_depth, tree_num, step_size) = tree_info
    total_feature_list = TA.get_gbdt_and_lr_feature(tree_leaf, tree_depth=tree_depth, tree_num=tree_num) #特征通过GBDT模型编码
    # 上面得到的特征编码是稀疏矩阵格式的，需要转换为便于计算的CSC格式
    result_list = np.dot(csc_matrix(mix_lr_coef), total_feature_list.tocsc().T).toarray()[0]
    sigmoid_ufunc = np.frompyfunc(sigmoid, 1, 1) # 可以将计算单个值的函数转换为一个能对数组中每个元素计算的 ufunc 函数。
    return sigmoid_ufunc(result_list)


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x))


def get_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print ("auc:%.5f" %(auc_score))


def get_accuary(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of test data
    """
    score_thr = 0.5
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuary_score = right_num/total_num
    print ("accuary:%.5f" %(accuary_score))


def run_check_core(test_feature, test_label, model, score_func):
    """
    Args:
        test_feature:
        test_label:
        model: tree model GBDT 模型实例
        score_func: use different model to predict
    """
    predict_list = score_func(test_feature, model)
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)

# GBDT 模型在测试数据上的表现测试
def run_check(test_file, tree_model_file, feature_num_file):
    """
    Args:
        test_file:file to test performance
        tree_model_file:gbdt model filt
        feature_num_file:file to store feature num
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    tree_model = xgb.Booster(model_file=tree_model_file) #加载 GBDT 模型
    run_check_core(test_feature, test_label, tree_model, predict_by_tree) #predict_by_tree:打分函数

def run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model, mix_lr_coef, tree_info, score_func):
    """
    core function to test the performance of the mix model
    Args:
        test_feature:
        test_label:
        mix_tree_model: GBDT树模型实例
        mix_lr_coef: LR模型参数列表
        tree_info:tree_depth, tree_num, step_size
        score_func: different score_func
    """
    predict_list =score_func(test_feature, mix_tree_model, mix_lr_coef, tree_info)
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)


# GBDT和LR混合模型在测试数据上的表现测试
def run_check_lr_gbdt(test_file, tree_mix_model_file, lr_coef_mix_model_file, feature_num_file):
    """
    Args:
        test_file:
        tree_mix_model_file: tree part of mix model
        lr_coef_mix_model_file:lr part of mix model
        feature_num_file:
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    mix_tree_model = xgb.Booster(model_file=tree_mix_model_file) #加载GBDT模型文件
    mix_lr_coef = np.genfromtxt(lr_coef_mix_model_file, dtype=np.float32, delimiter=",") #加载LR模型参数
    tree_info = TA.get_mix_model_tree_info()
    run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model, \
                           mix_lr_coef, tree_info, predict_by_lr_gbdt)


if __name__ == "__main__":
    # if len(sys.argv) == 4:
    #     test_file = sys.argv[1]
    #     tree_model = sys.argv[2]
    #     feature_num_file = sys.argv[3]
    #     run_check(test_file, tree_model, feature_num_file)
    # elif len(sys.argv) == 5:
    #     test_file = sys.argv[1]
    #     tree_mix_model = sys.argv[2]
    #     lr_coef_mix_model = sys.argv[3]
    #     feature_num_file = sys.argv[4]
    #     run_check_lr_gbdt(test_file, tree_mix_model, lr_coef_mix_model,  feature_num_file)
    # else:
    #     print ("check gbdt model usage: python xx.py test_file  tree_model feature_num_file")
    #     print ("check lr_gbdt model usage: python xx.py test_file tree_mix_model lr_coef_mix_model feature_num_file")
    #     sys.exit()
    # run_check("../data/test_file","../data/xgb.model", "../data/feature_num") # 测试代码：测试GBDT模型效果
    run_check_lr_gbdt("../data/test_file", "../data/xgb_mix_model", "../data/lr_coef_mix_model", "../dta/feature_num") #测试代码：测试GBDT和LR混合模型性能