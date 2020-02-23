#### -*-coding222:utf8-*-
"""
author:zhiyuan
date:20190316
train gbdt model
"""
import xgboost as xgb
import sys
sys.path.append("../")
import util.get_feature_num as GF
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix


def get_train_data(train_file, feature_num_file):
    """
    get train data and label for training
    """
    total_feature_num = GF.get_feature_num(feature_num_file)
    #最后一列为label
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    #读取前103列（为特征）
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= feature_list)
    return train_feature, train_label


def train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate):
    """
    Args:
        train_mat:train data and label
        tree_depth:
        tree_num:total tree num
        learning_rate: step_size
    Return:Booster
    """
    #优化目标：回归问题的线性优化
    #目标函数就是我们求最优划分特征时一阶偏导和二阶偏导所使用的函数
    para_dict = {"max_depth":tree_depth, "eta":learning_rate, "objective":"reg:linear","silent":1}
    bst = xgb.train(para_dict, train_mat, tree_num)
    # print(xgb.cv(para_dict,train_mat, tree_num, nfold=5,metrics={"auc"})) #交叉验证代码，看auc指标，这里怎么没有传入训练好的模型 bst ??
    #                                                                        # 最后一棵树的auc指标为0.915386，比LR模型的auc指标好
    return bst


#组合各种树深度/树个数/学习率
def choose_parameter():
    """
    Return:
         list: such as [(tree_depth, tree_num, step_size),...]
    """
    result_list = []
    tree_depth_list = [4, 5, 6]
    tree_num_list = [10, 50, 100]
    learning_rate_list = [0.3, 0.5, 0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))
    return result_list


def grid_search(train_mat):
    """
    Args:
        train_mat: train data and train label
    select the best parameter for training model 为GBDT模型选取最优的参数（树个数/树深度/学习率）
    """
    para_list = choose_parameter()
    # 统计各种参数时的AUC值
    for ele in para_list:
        (tree_depth, tree_num, learning_rate) = ele
        para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
        res = xgb.cv(para_dict, train_mat, tree_num, nfold=5, metrics={'auc'})
        auc_score = res.loc[tree_num-1, ['test-auc-mean']].values[0] #取最后一棵树的第一列值
        print ("tree_depth:%s,tree_num:%s, learning_rate:%s, auc:%f" \
              %(tree_depth, tree_num, learning_rate, auc_score))


def train_tree_model(train_file , feature_num_file, tree_model_file):
    """
    Args:
        train_file: data for train model
        tree_model_file: file to store model
        feature_num_file:file to record feature total num
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    # grid_search(train_mat) #尝试看看哪个参数组合最好（最终发现深度为6，树个数为10，学习率为0.3时，AUC最大，效果最好）
    tree_num = 10 #GBDT需要多少棵树
    tree_depth = 4 #GBDT树的深度
    learning_rate = 0.3 # fm = fm-1 + step_size*Tm, here step_size=learning_rate
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(tree_model_file)

#将GBDT编码的特征转化为LR模型需要的特征编码
def get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth):
    """
    Args:
        tree_leaf: prediction of the tree model GBDT预测的样本所属哪个叶子节点
        tree_num:total_tree_num
        tree_depth:total_tree_depth
    Return:
         Sparse Matrix to record total train feature for lr part of mixed model
         假设树深度为6，叶子节点数为2^6=64，一共有10棵树，特征就是64*10=640维，每一棵树
         对应特征编码，只有一个位置上是1，其余位置为0，所以这里使用稀疏矩阵存储能节省大量内存。
    """
    total_node_num = 2**(tree_depth + 1) - 1 #总节点数
    yezi_num = 2**tree_depth #叶子节点数
    feiyezi_num = total_node_num - yezi_num #非叶子节点数
    total_col_num = yezi_num*tree_num #特征总的维度数
    total_row_num = len(tree_leaf) #样本数
    col = [] #稀疏矩阵存储(COO格式）需要的数据结构
    row = []
    data = []
    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            yezi_index = fix_index - feiyezi_num
            yezi_index  = yezi_index if yezi_index >= 0 else 0 #如果小于0，说明样本输出在非叶子节点，是不完全训练的树
                                                                #此时编号初始化为0
            col.append(base_col_index + yezi_index) #深度为4，第一棵树占据特征向量的0~15维，第二可数占据特征向量的16~31维
            row.append(base_row_index)
            data.append(1)
            base_col_index += yezi_num
        base_row_index += 1
    total_feature_list = coo_matrix((data, (row,col)), shape=(total_row_num, total_col_num))
    return total_feature_list


def get_mix_model_tree_info():
    """
    tree info of mix model
    """
    tree_depth = 4
    tree_num = 10
    step_size = 0.3
    result = (tree_depth, tree_num, step_size)
    return result

#训练GBDT和LR混合模型
def train_tree_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """
    Args:
        train_file:file for training model（输入文件）
        feature_num_file:file to store total feature len（输入文件）
        mix_tree_model_file: tree part of the mix model（GBDT树模型文件，输入文件）
        mix_lr_model_file:lr part of the mix model（LR模型文件，输入文件）
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    (tree_depth, tree_num, learning_rate) = get_mix_model_tree_info()
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(mix_tree_model_file)
    tree_leaf = bst.predict(train_mat, pred_leaf=True) #预测样本最终落在哪个节点上
    # print(tree_leaf[0]) #测试代码，打印一个样本的输出结果：一行，10列，每列值为样本落在数的哪个叶子节点上
    #                    # 这里没有使用深度为6，而是深度为4，因为我们发现深度为6，有64个叶子节点，63个非叶子节点，
    #                    # 树被训练完全的棵数不是很多（最小叶子节点序号为63，而输出很多都在63以下，表明没有训练完全）。
    #                    # 另外如果使用6作为树深度，10棵数，LR特征向量维度就是640，而我们训练样本只有3万多条，也不满
    #                    # 足特征维度和样本数比值为1:100的比例要求。换成深度为4，输出的10列数，大部分在15级以上
    #                    # （节点从0开始计数的，序号为15的节点，就是第一个叶子节点的编号）
    # print(np.max(tree_leaf)) #测试代码，打印最大的索引（为30，最后一个叶子节点的编号）
    # sys.exit()       #测试代码
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    # LR模型的训练代码
    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, cv=5)\
        .fit(total_feature_list, train_label)
    scores = list(lr_clf.scores_.values())[0]
    print ("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print ("Accuracy:%f(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, scoring='roc_auc', cv=5).fit(
        total_feature_list, train_label)
    scores = list(lr_clf.scores_.values())[0]
    print ("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print ("AUC:%f,(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    fw = open(mix_lr_model_file, "w+")
    coef = lr_clf.coef_[0]
    fw.write(','.join([str(ele) for ele in coef]))


if __name__ == "__main__":

    # if len(sys.argv) == 4:
    #     train_file = sys.argv[1]
    #     feature_num_file = sys.argv[2]
    #     tree_model = sys.argv[3]
    #     train_tree_model(train_file, feature_num_file, tree_model)
    # elif len(sys.argv) == 5:
    #     train_file = sys.argv[1]
    #     feature_num_file = sys.argv[2]
    #     tree_mix_model = sys.argv[3]
    #     lr_coef_mix_model = sys.argv[4]
    #     train_tree_and_lr_model(train_file,  feature_num_file, tree_mix_model, lr_coef_mix_model)
    # else:
    #     print ("train gbdt model usage: python xx.py train_file feature_num_file tree_model")
    #     print ("train lr_gbdt model usage: python xx.py train_file feature_num_file tree_mix_model lr_coef_mix_model")
    #     sys.exit()
    # train_tree_model("../data/train_file", "../data/feature_num", "../data/xgb.model") # 测试代码
    train_tree_and_lr_model("../data/train_file", "../data/feature_num","../data/xgb_mix_model", "../data/lr_coef_mix_model") #测试代码
