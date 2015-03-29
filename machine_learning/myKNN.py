# -*- coding: utf-8 -*-

import math, sys

class MyKNN():

    def __init__(self):
        self.distances = []
        self.trainset = []


    def cosine_similarity(self, v1, v2):
        sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
        for i in range(0, len(v1)):
            sum_xx += math.pow(v1[i], 2)
            sum_xy += v1[i] * v2[i]
            sum_yy += math.pow(v2[i], 2)

        return sum_xy / math.sqrt(sum_xx * sum_yy)

    def train(self, trainset):
        self.trainset = trainset


    def predict(self, inputset, k):

        # calculate the distance between input data and train data
        for idx, data in enumerate(self.trainset):
            self.distances.append((data[0], cosine_similarity(data[1], input_tf)))
            print '\tTF(%d) = %f' % (idx, self.distances[-1][1])


        class_count = dict()
        print '(2) 取K個最近鄰居的分類, k = %d' % k
        for i, place in enumerate(sorted(self.distances, key=lambda x: x[1], reverse=True)):
            current_class = place[0]
            print '\tTF(%d) = %f, class = %s' % (i, place[1], place[0])
            class_count[current_class] = class_count.get(current_class, 0) + 1
            if (i + 1) >= k:
                break

        print '(3) K個最近鄰居分類出現頻率最高的分類當作最後分類'
        input_class = ''
        for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
            if i == 0:
                input_class = c
            print '\t%s, %d' % (c, class_count.get(c))
        print '(4) 分類結果 = %s' % input_class

        return input_class




def create_trainset():
    trainset_tf = dict()
    trainset_tf[u'C63發表會'] = (15, 25, 0, 5, 8, 3)
    trainset_tf[u'BMW i8'] = (35, 40, 1, 3, 3, 2)
    trainset_tf[u'林書豪'] = (5, 0, 35, 50, 0, 0)
    trainset_tf[u'湖人隊'] = (1, 5, 32, 15, 0, 0)
    trainset_tf[u'Android 5.0'] = (10, 5, 7, 0, 2, 30)
    trainset_tf[u'iPhone6'] = (5, 5, 5, 15, 8, 32)

    trainset_class = dict()
    trainset_class[u'C63發表會'] = 'P'
    trainset_class[u'BMW i8'] = 'P'
    trainset_class[u'林書豪'] = 'S'
    trainset_class[u'湖人隊'] = 'S'
    trainset_class[u'Android 5.0'] = 'T'
    trainset_class[u'iPhone6'] = 'T'

    return trainset_tf, trainset_class


def cosine_similarity(v1, v2):
    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
    for i in range(0, len(v1)):
        sum_xx += math.pow(v1[i], 2)
        sum_xy += v1[i] * v2[i]
        sum_yy += math.pow(v2[i], 2)

    return sum_xy / math.sqrt(sum_xx * sum_yy)


def knn_classify(input_tf, trainset_tf, trainset_class, k):
    tf_distance = dict()
    # 計算每個訓練集合特徵關鍵字字詞頻率向量和輸入向量的距離

    print '(1) 計算向量距離'
    for place in trainset_tf.keys():
        tf_distance[place] = cosine_similarity(trainset_tf.get(place), input_tf)
        print '\tTF(%s) = %f' % (place, tf_distance.get(place))

    # 把距離排序，取出k個最近距離的分類

    class_count = dict()
    print '(2) 取K個最近鄰居的分類, k = %d' % k
    for i, place in enumerate(sorted(tf_distance, key=tf_distance.get, reverse=True)):
        current_class = trainset_class.get(place)
        print '\tTF(%s) = %f, class = %s' % (place, tf_distance.get(place), current_class)
        class_count[current_class] = class_count.get(current_class, 0) + 1
        if (i + 1) >= k:
            break

    print '(3) K個最近鄰居分類出現頻率最高的分類當作最後分類'
    input_class = ''
    for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
        if i == 0:
            input_class = c
        print '\t%s, %d' % (c, class_count.get(c))
    print '(4) 分類結果 = %s' % input_class


if __name__ == '__main__':

    inputFile = sys.argv[1]

    with open(inputFile) as f:
        dataset = f.readlines()
        dataset = [s[:-1].split(',') for s in dataset]
        dataset = [map(lambda s : float(s), x) for x in dataset]
        dataset = [(d[-1], d[:-2]) for d in dataset]
    dataset = dataset[-100:]

    target = [1.00554, 6.683276]




    trainset_tf = [
            (1, [15, 25, 0, 5, 8, 3]),
            (1, [35, 40, 1, 3, 3, 2]),
            (2, [5, 0, 35, 50, 0, 0]),
            (2, [1, 5, 32, 15, 0, 0]),
            (3, [10, 5, 7, 0, 2, 30]),
            (3, [5, 5, 5, 15, 8, 32])
    ]

    input_tf = (10, 2, 50, 56, 8, 5)

    myKnn = MyKNN()
    myKnn.train(dataset)
    myKnn.predict(target, 10)
