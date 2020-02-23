#-*-coding:utf8-*-
"""
mat util for personal rank algo
"""
#1������user-item����ͼ�õ�M����
#2��(E-alpha*MT)-1
# �������ģ���ֹ������Ϊ��������
from __future__ import division
from scipy.sparse import coo_matrix
import numpy as np
import util.read as read
import sys
def graph_to_m(graph):
    """
    :param graph: user item graph
    :return:  a coo_matrix, sparse mat M
              a list, total user item point
              a dict, map all the point to row index
    """
    vertex = list(graph.keys())
    address_dict = {}
    total_len = len(vertex)
    for index in range(len(vertex)):
        address_dict[vertex[index]] = index
    # coo ϡ�����洢��Ҫ��3�����ݽṹ
    row = []
    col = []
    data = []
    for element_i in graph:
        #M������M+N��*M+N�еľ����а��������нڵ㡢�а��������нڵ㣬����ת�ƾ���
        # ������ֵ�������Ե�һ�е�2�о����������һ�ж�Ӧ�����е��ڶ��ж�Ӧ��������ӣ���M12ֵΪ��һ�ж�����ȵĵĵ�����
        # �����һ�ж�Ӧ�����е��ڶ��ж�Ӧ����û�����ӱߣ�M12ֵΪ0��
        weight = round(1/len(graph[element_i]), 3)
        row_index = address_dict[element_i]
        for element_j in graph[element_i]:
            col_index = address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data,(row, col)), shape=(total_len, total_len))
    return m, vertex, address_dict

def mat_all_point(m_mat, vertex, alpha):
    """
    get E-alpha*m_mat.T
    :param m_mat:
    :param vertex: total item and user point
    :param alpha: the prob for random walking
    :return: a sparse matrix
    """
    # np.eye() #���׳��ڴ�
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    # print(eye_t.todense())
    # sys.exit()
    # ���� tocsr() ��������
    return eye_t.tocsr() - alpha * m_mat.tocsr().transpose()

if __name__ == "__main__":
    graph = read.get_graph_from_data("../data/log.txt")
    m,vertex, address_dict = graph_to_m(graph)
    # print(address_dict)
    # print(m.todense())
    m = mat_all_point(m, vertex, 0.8)
    print(m.todense())


