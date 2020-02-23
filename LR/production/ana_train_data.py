# -*-coding:utf8-*-
"""
author:zhiyuan
date:20190310
feature selection and data selection
"""
import pandas as pd #快速处理行索引和列索引
import numpy as np
import operator
import sys


def get_input(input_train_file, input_test_file):
    """
    Args:
        input_train_file:
        input_test_file:
    Return:
         pd.DataFrame train_data
         pd.DataFrame test_data
    """
    #
    dtype_dict = {"age": np.int32,
                  "education-num": np.int32,
                  "capital - gain": np.int32,
                  "capital - loss": np.int32,
                  "hours - per - week": np.int32}
    use_list = list(range(15))
    # fnlwgt 列（ID列）不需要
    del use_list[2]
    # use_list.remove(2)
    # read_csv() 默认字段类型为字符串，如果要是其他类型，需要在dtype参数里指定
    train_data_df = pd.read_csv(input_train_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    #有?(na)字段的行都不要（axis=0，表示删除包含缺省值的行）how=any表示只要有一个字段为na，就删除
    train_data_df = train_data_df.dropna(axis=0, how="any")
    #看看样本选择前后各有多少行
    # print(train_data_df.shape)
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how="any")
    return train_data_df, test_data_df


def label_trans(x):
    """
    Args:
        x: each element in fix col of df
    """
    if x == "<=50K":
        return "0"
    if x == ">50K":
        return "1"
    return "0"

#离散特征处理
def process_label_feature(lable_feature_str, df_in):
    """
    Args:
        lable_feature_str:"label"
        df_in:DataFrameIn
    """
    # 对名为 lable_feature_str 列的 dataframe 作 label_trans 转换
    df_in.loc[:, lable_feature_str] = df_in.loc[:, lable_feature_str].apply(label_trans)


def dict_trans(dict_in):
    """
    Args:
        dict_in: key str（属性值）, value int（样本数）
    Return:
        a dict, key str, value index for example 0,1,2
    """
    output_dict = {}
    index = 0
    # 按照 value 排序
    for zuhe in sorted(dict_in.items(), key = operator.itemgetter(1), reverse= True):
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict

#离散特征离散化处理函数
def dis_to_feature(x, feature_dict):
    """
    Args:
        x: element
        feature_dict: pos dict
    Return:
        a str as "0,1,0" #只会有一个数是1，其他全部为0
    """
    output_list = [0] * len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])

# 离散特征的离散化
# 举例：系统中有中国、美国、日本（顺序的），这样国家是中国的样本，离散化后的值为[1,0,0]。
def process_dis_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process dis feature for lr train
    """
    #存储每个不同的值，以及对应的样本数
    origin_dict = df_train.loc[:, feature_str].value_counts().to_dict()
    # feature_dict 存储枚举值所在的位置序号
    feature_dict = dict_trans(origin_dict)
    #测试数据/训练数据集合的 feature_str 列都需要做转换
    df_train.loc[:, feature_str] = df_train.loc[:,feature_str].apply(dis_to_feature, args= (feature_dict, ))
    df_test.loc[:, feature_str] = df_test.loc[:,feature_str].apply(dis_to_feature, args= (feature_dict, ))
    # print(df_train。loc[:3, feature_str]) #打印 df_train 的 feature_str 列的前4行

    return len(feature_dict)

#连续型特征离散化使用的函数
def list_trans(input_dict):
    """
    Args:
        input_dict:{'count': 30162.0, 'std': 13.134664776855985, 'min': 17.0, 'max': 90.0, '50%': 37.0,
                    '25%': 28.0, '75%': 47.0, 'mean': 38.437901995888865}
    Return:
         a list, [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    output_list = [0]*5
    key_list = ["min", "25%","50%","75%","max"]
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in input_dict:
            print ("error")
            sys.exit()
        else:
            output_list[index] = input_dict[fix_key]
    return output_list

#连续型特征的离散化
# 举例：每周工作23小时，统计样本中这一特征值的分布，比如需要4段离散化：
# 每周工作0~18小时的样本占25%，18~25小时的样本占25%，25~40小时的样本占25%，40小时以上占25%，
# 这样，每周工作23小时的样本，它在第2个区间，离散化为[0,1,0,0]：
def con_to_feature(x, feature_list):
    """
    Args:
        x: element
        feature_list: list for feature trans
    Return:
        str, "1_0_0_0"
    """
    feature_len = len(feature_list) -1
    result = [0] * feature_len
    for index in range(feature_len):
        if x >= feature_list[index] and x <= feature_list[index + 1]:
            result[index] = 1
            #找到连续值所在的区间序号，置1，其他位置0，然后返回
            return ",".join([str(ele) for ele in result])
    return ",".join([str(ele) for ele in result])


def process_con_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process con feature for lr train
    """
    # describe() 可以得到1/4、1/2、3/4分位点值
    origin_dict = df_train.loc[:, feature_str].describe().to_dict()
    feature_list = list_trans(origin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(con_to_feature, args=(feature_list, ))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(con_to_feature, args=(feature_list, ))
    # print( df_test.loc[:3, feature_str]) #打印前4条记录的feature_str列
    # print(feature_list)
    return len(feature_list) -1


def output_file(df_in, out_file):
    """

    write data of dataframe(df_in) to out_file
    """
    fw = open(out_file, "w+")
    for row_index in df_in.index:
        outline = ",".join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline + "\n")
    fw.close()

#组合特征的取值方法
def add(str_one, str_two):
    """
    Args:
        str_one:"0,0,1,0"
        str_two:"1,0,0,0"
    Return:
        str such as"0,0,1,0,0"
    """
    list_one = str_one.split(",")
    list_two = str_two.split(",")
    list_one_len = len(list_one)
    list_two_len = len(list_two)
    return_list = [0]*(list_one_len*list_two_len)
    try:
        index_one = list_one.index("1")
    except:
        index_one = 0
    try:
        index_two = list_two.index("1")
    except:
        index_two = 0
    return_list[index_one*list_two_len + index_two] = 1
    return ",".join([str(ele) for ele in return_list])

# 特征组合的函数
def combine_feature(feature_one, feature_two, new_feature, train_data_df, test_data_df, feature_num_dict):
    """
    Args:
        feature_one: 第一个特征的名称
        feature_two: 第二个特征的名称
        new_feature: combine feature name 组合特征名称
        train_data_df: 训练数据的dataframe
        test_data_df: 测试数据的dataframe
        feature_num_dict: ndim of every feature, key:feature name, value: len of the dim（一个特征经过离散化后得到的离散值列表）
    Return:
        new_feature_num
    """
    # pandas dataframe 中 axis=1 表示沿水平方向计算 axis=0 表示沿垂直方向计算
    train_data_df[new_feature] = train_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    test_data_df[new_feature] = test_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    if feature_one not in feature_num_dict:
        print ("error")
        sys.exit()
    if feature_two not in feature_num_dict:
        print ("error")
        sys.exit()
    return feature_num_dict[feature_one]*feature_num_dict[feature_two]

def ana_train_data(input_train_data, input_test_data, out_train_file, out_test_file, feature_num_file):
    """
    Args:
        input_train_data:
        input_test_data:
        out_train_file:
        out_test_file:
        feature_num_file:
    """
    train_data_df, test_data_df = get_input(input_train_data, input_test_data)
    label_feature_str = "label"
    #离散特征
    dis_feature_list = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]
    #连续特征
    con_feature_list = ["age","education-num","capital-gain","capital-loss","hours-per-week"]
    index_list = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    process_label_feature(label_feature_str, train_data_df)
    process_label_feature(label_feature_str, test_data_df)
    dis_feature_num = 0
    con_feature_num = 0
    feature_num_dict = {} #组合特征
    for dis_feature in dis_feature_list:
        tmp_feature_num = process_dis_feature(dis_feature, train_data_df, test_data_df)
        dis_feature_num += tmp_feature_num
        feature_num_dict[dis_feature] = tmp_feature_num
    for con_feature in con_feature_list:
        tmp_feature_num = process_con_feature(con_feature, train_data_df, test_data_df) #一个特征值经过离散化处理后得到的离散值列表长度
        con_feature_num += tmp_feature_num
        feature_num_dict[con_feature] = tmp_feature_num
    #将年龄和收入两个特征组合，新的组合特征名为age_gain
    new_feature_len = combine_feature("age", "capital-gain", "age_gain", train_data_df, test_data_df, feature_num_dict)
    # print(train_data_df['age'][:2])
    # print(train_data_df['capital-gain'][:2])
    # print(train_data_df['age_gain'][:2])
    # sys.exit(1)
    new_feature_len_two = combine_feature("capital-gain", "capital-loss", "loss_gain", train_data_df, test_data_df, feature_num_dict)
    # 将label列调整到最后，loss_gain调整为倒数第二列，age_gain调整为倒数第3列
    train_data_df = train_data_df.reindex(columns=index_list +["age_gain","loss_gain","label"])
    test_data_df = test_data_df.reindex(columns=index_list +["age_gain", "loss_gain", "label"])
    output_file(train_data_df, out_train_file)
    output_file(test_data_df, out_test_file)
    fw = open(feature_num_file, "w+") #特征数写入文件
    fw.write("feature_num=" + str(dis_feature_num + con_feature_num+new_feature_len+new_feature_len_two) )


if __name__ == "__main__":
    ana_train_data("../data/train.txt", "../data/test.txt", "../data/train_file", "../data/test_file", "") #测试代码
    if len(sys.argv) < 6:
        print ("usage: python xx.py origin_train origin_test train_file test_file feature_num_file")
        sys.exit()
    else:
        origin_train = sys.argv[1]
        origin_test = sys.argv[2]
        train_file = sys.argv[3]
        test_file = sys.argv[4]
        feature_num_file = sys.argv[5]
        ana_train_data(origin_train, origin_test, train_file, test_file, feature_num_file)
        #ana_train_data("../data/train.txt", "../data/test.txt", "../data/train_file", "../data/test_file", "../data/feature_num")
