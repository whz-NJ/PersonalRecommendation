# -*-coding:utf8-*-
"""
author:zhiyuan
date:20190310
use lr model to check the performance in test file 使用训练好的LR模型，在测试数据里检验效果
准确率：
假设样本实际label为   1 0 1 0 1
通过LR模型估计label为 0 1 0 0 1
此时准确率为：2/5
AUC 表明了模型对样本预测的一种序关系（在点击率预测中，我们不是判断具体样本是正样本还是负样本，而是只要得出样本的概率大小的排序，
                                      取排序靠前的topN推荐即可，所以点击率预估中，AUC指标更好地反应模型的性能）
label 概率(模型预测为1的概率)
1     0.9
1     0.8
1     0.3
0     0.2
0     0.4
上面正负样本对数=3*2=6（做AUC得分的分母），其中第一个正样本概率比两个负样本概率都高，所以第一个样本得分=2
同理第二个正样本得分也是2，第三个正样本得分为1，所以AUC=(2+2+1)/6=5/6。
工业界认为，AUC得分在0.7以上的模型便可以上线应用。
"""
from __future__ import division
import numpy as np
from sklearn.externals import joblib
import math
import sys
sys.path.append("../")
# from util import get_feature_num as gf
import util.get_feature_num as gf

def get_test_data(test_file, feature_num_file):
    """
    Args:
        test_file:file to check performance 用来检查模型性能的测试文件
        feature_num_file: the file record total num of feature（当前是118）
    Return:
         two np array: test _feature, test_label
    """
    # total_feature_num = gf.get_feature_num(feature_num_file)
    total_feature_num = 118 #测试代码
    # 通过np模块读入测试文件内容（label是测试文件的最后一列，所有usercols=-1）
    test_label = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= feature_list)
    return test_feature, test_label


def predict_by_lr_model(test_feature, lr_model):
    """
    predict by lr_model （调用 sklearn 实例方法）
    """
    result_list = [] #存储每个样本label为1的概率
    prob_list = lr_model.predict_proba(test_feature)
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1]) #下标为0的对应label为0的概率，下标为1的对应label为1的概率
    return result_list


def predict_by_lr_coef(test_feature, lr_coef):
    """
    predict by lr_coef（模型参数列表）
    """
    sigmoid_func = np.frompyfunc(sigmoid, 1, 1) #universal function，可以对np.array里的每个元素执行 sigmoid
                                                #一个输入，一个输出
    return sigmoid_func(np.dot(test_feature, lr_coef))


def sigmoid(x):  #阶跃函数
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x)) #e的-x次幂


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
    #按照模型的得分，对total_list排序
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
        predict_list: model predict score list 模型预测样本label为1的概率列表
        test_label: label of test data
    """
    score_thr = 0.5 #模型预测Label为1的概率大于0.5，就认为是正样本
                    #取0.5时，通过lr_coef（参数列表）或 lr_model（模式实例）判断样本的准确率不一样
                    # lr_coef的准确率要低一些，但两个模型的AUC值是一样的，由于我们不需要判断正负样本，
                   # 只需要取label为1的概率高的前topN推荐给用户即可，所以两个模型在点击率推荐场景效果一样
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
        model: lr_coef（参数列表）或 lr_model（模式实例）
        score_func: use different model to predict（一种是需要我们自己用模型参数和各个样本对应属性相乘再连加、
                                                    一种是直接调用 sklearn 模型的方法）
    """
    predict_list = score_func(test_feature, model) #预测label为1的概率
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)


def run_check(test_file, lr_coef_file, lr_model_file, feature_num_file):
    """
    Args:
        test_file: file to check performace 测试文件
        lr_coef_file: w1,w2 保存模型各个参数的文件
        lr_model_file: dump file 训练得到的模型对象整体导出的文件
        feature_num_file: file to record num of feature
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    lr_coef = np.genfromtxt(lr_coef_file, dtype=np.float32, delimiter=",")
    # 模型实例文件通过 sklearn 导入
    lr_model = joblib.load(lr_model_file)
    run_check_core(test_feature, test_label, lr_model, predict_by_lr_model)
    run_check_core(test_feature, test_label, lr_coef, predict_by_lr_coef)


if __name__ == "__main__":
    run_check("../data/test_file", "../data/lr_coef", "../data/lr_model_file", "../data/feature_num") #测试代码
    if len(sys.argv) < 5:
        print ("usage: python xx.py test_file coef_file model_file feature_num_file")
        sys.exit()
    else:
        test_file = sys.argv[1]
        coef_file = sys.argv[2]
        model_file = sys.argv[3]
        feature_num_file = sys.argv[4]
        run_check(test_file, coef_file, model_file, feature_num_file)

