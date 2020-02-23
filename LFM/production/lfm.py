#-*-coding:utf8-*-
"""
lfm model traing main function
"""
import operator
import sys

import numpy as np

sys.path.append("../util")

import util.read as read

def lfm_train(train_data, F, alpha, beta, step):
    """
    :param train_data: for lfm
    :param F: user vector len, item vector len
    :param alpha: regularization factor
    :param beta: leaning rate
    :param step: ierration num
    :return:
        dict: key: itemid, value:np.ndarray
        dict key userid, value:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        for data_instance in train_data:
            userid, itemid, label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
        delta = label - model_predict(user_vec[userid], item_vec[itemid])
        for index in range(F):
            # 这里删除2倍乘积，通过学习率beta一并代替
            user_vec[userid][index] += beta*(delta*item_vec[itemid][index] - alpha*user_vec[userid][index])
            item_vec[itemid][index] += beta*(delta*user_vec[userid][index] - alpha*item_vec[itemid][index])
        #接近收敛时，让参数变化慢一些
        beta = beta * 0.9
        return user_vec, item_vec
def init_model(vector_len):
    """
    使用标准的正态分布初始化
    :param vector_len: the len of vector
    :return: a ndarray
    """
    return np.random.randn(vector_len)

def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance（推荐强度）
    :param user_vector: model produce user vector
    :param item_vector: model produce item vector
    :return: a num
    """
    ## cos distance
    # res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    # res = np.sqrt(np.sum(np.square(user_vector - item_vector)))
    res = np.linalg.norm(user_vector - item_vector)
    return res
def model_train_process():
    """
    test lfm model train
    :return:
    """
    train_data = read.get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01,0.1, 50)
    # print(user_vec["1"])
    # print(item_vec['2455'])
    recom_result = give_recom_result(user_vec, item_vec, "24")
    ana_recom_result(train_data, "24", recom_result)

    # 通过查找redis，给出基于用户历史点击行为的推荐商品
    # for userid in user_vec:
    #     recom_result = give_recom_result(user_vec, item_vec, userid)
    #     #store recom_result in redis

    # 通过查redis，推荐与用户当前点击商品近似的商品
    # fix_num = 10
    # for itemid in item_vec:
    #     item_vector = item_vec[itemid]
    #     record = {}
    #     for itemid2 in item_vec:
    #         if itemid != itemid2:
    #             item_vector2 = item_vec[itemid2]
    #             # 欧式距离
    #             res = np.dot(item_vector, item_vector2) / (np.linalg.norm(item_vector) / np.linalg.norm(item_vector2))
    #             record[itemid] = res
    #     for zuhe in sorted(record.items(), key=operator.itemgetter(1), reverse=True)[:fix_num]:
    #         itemid0 = zuhe[0]
    #         # store itemid:(itemid0...) in redis, and when a user click itemid, we recommand (itemid0....)

def give_recom_result(user_vec, item_vec, userid):
    """
    use lfm model result give fix userid recom result
    :param user_vec:
    :param item_vec:
    :param userid:
    :return: a list[(itemid, score), (itemid2, score2)...]
    """
    fix_num = 10
    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec:
        item_vector = item_vec[itemid]
        # 欧式距离
        # res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)/np.linalg.norm(item_vector))
        #res = np.sqrt(np.sum(np.square(user_vector - item_vector)))
        res = np.linalg.norm(user_vector-item_vector)
        record[itemid] = res
    for zuhe in sorted(record.items(), key= operator.itemgetter(1), reverse=True)[:fix_num]:
        itemid = zuhe[0]
        score = round(zuhe[1], 3)
        recom_list.append((itemid, score))
    return recom_list

def ana_recom_result(train_data, userid, recom_list):
    """
    debug recom result for userid
    :param train_data: train data for lfm model
    :param userid:
    :param recom_list: result by lfm
    :return:
    """
    item_info = read.get_item_info("../data/movies.txt")
    for data_instance in train_data:
        tmp_userid,itemid,label = data_instance
        if tmp_userid == userid and label == 1:
            print(item_info[itemid])
    print("recom result")
    for zuhe in recom_list:
        print(item_info[zuhe[0]])

if __name__ == "__main__":
    model_train_process()
