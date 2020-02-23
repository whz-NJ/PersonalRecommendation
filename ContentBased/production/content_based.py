#-*-coding222:utf8-*-
"""
author:zhiyuan
date:2019
get up and online recommendation
"""

from __future__ import division
import os
import operator
import  sys
sys.path.append("../")
import util.read as read

#获取用户最喜欢的topk个类别，以及喜好程度得分
def get_up(item_cate, input_file):
    """
    Args:
        item_cate: key itemid, value: dict , key category value ratio（ratio值等于 1/item所属分类总数）
        input_file:user rating file
    Return:
        a dict: key userid, value [(category, ratio), (category1, ratio1)]
    """
    if not os.path.exists(input_file):
        return {}
    record = {}
    up = {}
    linenum = 0
    score_thr = 4.0
    topk = 2
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating, timestamp = item[0], item[1], float(item[2]), int(item[3])
        if rating < score_thr:
            continue
        if itemid not in item_cate:
            continue
        time_score = get_time_score(timestamp)
        if userid not in record:
            record[userid] = {}
        for fix_cate in item_cate[itemid]:
            if fix_cate not in record[userid]:
                record[userid][fix_cate] = 0
            #用户对某一分类的兴趣得分=用户对属于该分类的一个电影的评分*时间系数*该电影属于这个分类的比例
            record[userid][fix_cate] += rating * time_score * item_cate[itemid][fix_cate]
    fp.close()
    for userid in record:
        if userid not in up:
            up[userid] = []
        total_score = 0
        #将用户对于某种类别喜好程度得分从高到低排序，取topk个类别
        for zuhe in sorted(record[userid].items(), key = operator.itemgetter(1), reverse=True)[:topk]:
            # up[userid] 值为[(category, ratio), (category1, ratio1)]
            up[userid].append((zuhe[0], zuhe[1]))
            total_score += zuhe[1]
        #再把用户对某种类别喜好得分归一化处理
        for index in range(len(up[userid])):
            up[userid][index] = (up[userid][index][0], round(up[userid][index][1]/total_score, 3))
    return up

# 获取时间得分
def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp（用户评分时的时间戳）
    Return:
        time score
    """
    fix_time_stamp = 1476086345
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/100
    return round(1/(1+delta), 3)

#在线推荐函数
def recom(cate_item_sort, up, userid, topk= 10):
    """
    Args:
        cate_item_sort:reverse sort
        up:user profile
        userid:fix userid to recom
        topk:recom num
    Return:
         a dict, key userid value [itemid1, itemid2]
    """

    if userid not in up:
        return {}
    recom_result = {}
    if userid not in recom_result:
        recom_result[userid] = []
    for zuhe in up[userid]:
        cate = zuhe[0]
        ratio = zuhe[1]
        # 对用户感兴趣的类别商品进行召回，召回数目是topk*用户对这类别喜好程度比例权重
        num = int(topk*ratio) + 1
        if cate not in cate_item_sort:
            continue
        #cate_item_sort[cate]里保存的是cate分类的按照用户评分由高到低的itemId列表，这里取前num个itemId
        recom_list = cate_item_sort[cate][:num]
        recom_result[userid] += recom_list
    return  recom_result


def run_main():
    ave_score = read.get_ave_score("../data/ratings.txt")
    item_cate, cate_item_sort =read.get_item_cate(ave_score, "../data/movies.txt")
    up = get_up(item_cate, "../data/ratings.txt")
    print(len(up)) #测试代码
    print(up["1"]) #测试代码
    print(recom(cate_item_sort, up, "1"))

if __name__ == "__main__":
    run_main()
