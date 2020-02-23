
"""
author:zhiyuan
date:20190310
train lr model
"""
import sys
sys.path.append("../")
from sklearn.linear_model import LogisticRegressionCV as lrcv #使用带交叉验证的LR
from sklearn.externals import joblib
import util as gf
import numpy as np


def train_lr_model(train_file, model_coef, model_file, feature_num_file):
    """
    Args:
        train_file: process file for lr train(处理好的训练文件)
        model_coef: w1 w2...(lr模型参数)
        model_file:model pkl(lr模型实例化输出)
        feature_num_file: file to record num of feature
    """
    # total_feature_num = gf.get_feature_num(feature_num_file)
    total_feature_num = 118 # 测试代码
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= -1) # label是最后一列
    feature_list = range(total_feature_num) #118列的特征（包括连续和离散的特征）
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= feature_list) #读入118维的特征
    # 这里的正则化参数alpha为1（可以有多个，取各个数的倒数），这里使用L2正则化（倾向于将各特征对应参数学小，而不是为0）
    # 参数迭代停止的条件：0.0001，最大迭代次数500次，cv=5表示将测试数据分成5份，每份拿20%作为测试，拿80%作为训练，一共训练5次
    # 还有一个参数 solver ：可以选牛顿法(默认)、随机梯度下降法，L2正则化时只能用牛顿或随机梯度下降法
    lr_cf = lrcv(Cs=[1,10,100], penalty="l2", tol=0.0001, max_iter=500, cv=5).fit(train_feature, train_label)
    # lr_cf = lrcv(Cs=[1], penalty="l2", tol=0.0001, max_iter=500, cv=5).fit(train_feature, train_label)
    scores = list(lr_cf.scores_.values())[0] #这里scores是5行（对应5次训练）3列的矩阵（对应Cs正则化参数个数为3个）
    print ("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis = 0)]))) # 打印各个Cs参数对应的准确率（越大越好）
    # 正态分布里，均值+/- 两倍标准差能涵盖90%以上的样本
    print ("Accuracy:%s (+-%0.2f)" %(scores.mean(), scores.std()*2)) #三个Cs正则化参数对应的准确率平均值，以及标准差值
    lr_cf = lrcv(Cs=[1], penalty="l2", tol=0.0001, max_iter=500, cv=5, scoring="roc_auc").fit(train_feature, train_label)
    scores = list(lr_cf.scores_.values())[0]
    #AUC准确率更能反映模型效果
    print ("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)]))) #不同Cs参数的AUC准确率
    print ("AUC:%s (+-%0.2f)" %(scores.mean(), scores.std()*2)) # AUC均值，以及标准差值
    coef = lr_cf.coef_[0] #将得到的模型参数输出
    fw = open(model_coef, "w+")
    fw.write(",".join(str(ele) for ele in coef))
    fw.close()
    joblib.dump(lr_cf, model_file) #模型实例保存到文件


if __name__ == "__main__":
    train_lr_model("../data/train_file", "../data/lr_coef","../data/lr_model_file", "") #测试代码
    if len(sys.argv) < 5:
        print ("usage: python xx.py train_file coef_file model_file featuren_num_file")
        sys.exit()
    else:
        train_file = sys.argv[1]
        coef_file = sys.argv[2]
        model_file = sys.argv[3]
        feature_num_file = sys.argv[4]
        train_lr_model(train_file, coef_file, model_file, feature_num_file)

        #train_lr_model("../data/train_file", "../data/lr_coef", "../data/lr_model_file", "../data/feature_num")