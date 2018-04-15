# -*- coding:utf-8 -*-

# DBSCAN思路
from numpy import *
import matplotlib.pyplot as plt

# 参考：https://blog.csdn.net/u014028027/article/details/72185796
data = '''
1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459'''

# 两个点的欧拉距离
def Elu_Distance(a, b):
    dist = sqrt(sum(square(a-b)))
    return dist

# 加载样本点
def load_DataSet(data):
    data_array = data.strip().split(',')
    dataSet = [(float(data_array[i]), float(data_array[i+1])) for i in range(1, len(data_array)-1,3)]
    return dataSet

# 实现DBSCAN算法
# 大致思想如下：
#
# 初始化核心对象集合cores为空，遍历一遍样本集DataSet中所有的样本，
#    计算每个样本点的ε-邻域中包含样本的个数，如果个数大于等于min_point，
#    则将该样本点加入到核心对象集合中。初始化聚类簇数k = 0， 初始化未访问样本集和为P = D。
# 当T集合中存在样本时执行如下步骤：
# 2.1记录当前未访问集合P_old = P
# 2.2从cores中随机选一个核心对象o,初始化一个队列coreQ = [core]
# 2.3P = P-o(从T中删除o)
# 2.4当Q中存在样本时执行：
# 2.4.1取出队列中的首个样本q
# 2.4.2计算q的ε-邻域中包含样本的个数，如果大于等于MinPts，则令S为q的ε-邻域与P的交集，
# coreQ = coreQ+S, not_visit = not_visit-S
# 2.5 k = k + 1,生成聚类簇为Ck = not_visit_old - not_visit
# 2.6 T = T - Ck
# 划分为C= {C1, C2, ……, Ck}


def dbscan(dataSet, e, min_point):
    k = 0    # 聚类个数
    cores = set()   # 核心对象集合
    not_visit = set(dataSet)  # 未访问对象
    #print(not_visit)
    clusters = []  # 簇对象
    for di in dataSet:
        if len([dj for dj in dataSet if Elu_Distance(array(di), array(dj))<=e])>=min_point:
            cores.add(di)   # 将核心点加入到核心集合中

    while len(cores):
        not_visit_old = not_visit
        # 从cores中随机选一个核心对象core,初始化一个队列coreQ = [core]
        core = list(cores)[random.randint(0, len(cores))]
        # not_visit = not_visit - o(从T中删除o)
        not_visit = not_visit - set(core)
        coreQ = []
        coreQ.append(core)
        #print(len(coreQ))
        while len(coreQ):
            coreq = coreQ[0]
            # 将以core点为核心的所有样本点找出来
            core_samples = [di for di in dataSet if Elu_Distance(array(di), array(coreq)) <= e]
            if len(core_samples) >= min_point:
                # 计算q的ε - 邻域中包含样本的个数，如果大于等于MinPts，则令S为q的ε - 邻域与not_visit的交集
                # not_visit中的个数会逐渐减少,最后S就会变成空，从而coreQ会逐渐变成空队列
                S = not_visit & set(core_samples)
                coreQ += list(S)
                not_visit = not_visit - S
            coreQ.remove(coreq)
        k += 1
        Ck = not_visit_old - not_visit
        cores = cores - Ck
        clusters.append(list(Ck))
    return clusters

# 画图，将样本点放在二维平面进行展示，最后以颜色区分来聚类
def draw(clusters):
    colors = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for i in range(len(clusters)):
        x = []
        y = []
        for j in range(len(clusters)):
            x.append(clusters[i][j][0])
            y.append(clusters[i][j][1])
        plt.scatter(x, y, marker='x', color=colors[i])
    plt.show()



if __name__ == '__main__':
    dataSet = load_DataSet(data)
    clusters = dbscan(dataSet, 0.11, 5)
    print(clusters, shape(clusters))
    draw(clusters)