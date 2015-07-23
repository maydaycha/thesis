# -*- coding: utf-8 -*-

# import test
# test.Test().testing()
# test.t1()
from __future__ import division
from sklearn.naive_bayes import GaussianNB

import numpy as np
import arff
import math
import sys

levels = ['normal', 'warning', 'drift']

class DriftDetiection():

    def __init__(self):
        self.min_standard_deviation = 0
        self.min_error_rate = 0
        self.current_min = 0
        self.levels = ['normal', 'warning', 'drift']

    def detect(self, predict_labels, real_labels, k):
        print "K: %d" % k
        miss_label_count = (predict_labels != real_labels).sum()
        total_instance = len(predict_labels)
        precision_rate = (total_instance - miss_label_count) / total_instance
        miss_label_rate = miss_label_count / total_instance

        print "miss label: %f" % miss_label_rate

        # Binomial Distribution
        # 至少 k 個分錯的機率
        # P(Sn >= k) = 1 - ( P(Sn = 0) + P(Sn = 1)  + ..... + P(Sn = k-1) )
        error_rate = 0.0
        for i in range(k):
            # print "1: %f" % (math.factorial(total_instance) / (math.factorial(i) * math.factorial(total_instance - i)))
            # print "2: %f" % math.pow(miss_label_rate, i)
            # print "3：%f" % math.pow((1 - miss_label_rate), (total_instance - 1))
            # print "3：%f, %f" % ((1 - miss_label_rate), (total_instance - 1))
            error_rate += (math.factorial(total_instance) / (math.factorial(i) * math.factorial(total_instance - i))) * math.pow(miss_label_rate, i) * math.pow((1 - miss_label_rate), (total_instance - 1))
            # print "error rate: %f" % error_rate
        error_rate = 1 - error_rate
        print "error rate: %f" % error_rate

        # standard deviation => s = math.sqrt(Pi * (1-Pi) / i)
        standard_deviation = math.sqrt(error_rate * (1 - error_rate) / total_instance)


        if (error_rate + standard_deviation) <= self.current_min:
            self.min_error_rate = error_rate
            self.min_standard_deviation = standard_deviation
        print "current min: %f " % self.current_min
        print "min_error_rate: %f" % self.min_error_rate
        print "min_standard_deviation: %f" % self.min_standard_deviation

        # detect the level of concept drift
        if (error_rate + standard_deviation) >= (self.min_error_rate + 3 * self.min_standard_deviation):
            return self.levels[2]
        elif (error_rate + standard_deviation) >= (self.min_error_rate + 2 * self.min_standard_deviation):
            return self.levels[1]
        else:
            return self.levels[0]







if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: main.py <split size>"
        exit(-1)
    # load dataset
    dataset = arff.load(open('../data/concept_drift_data/usenet1.arff', 'rb'))

    data = list(d[:-1] for d in dataset['data'])

    # convert string to integer
    data = [map(int, x) for x in data]
    # data = np.array(data)

    target = list((int(c[-1]) for c in dataset['data']))
    # target = np.array(target)

    # split data
    split_data = []
    split_target = []
    split_size = int(sys.argv[1])

    for i in range(split_size):
        from_index = int(i * len(data) / split_size)
        to_index = int((i+1) * len(data) / split_size - 1)
        split_data.append(data[from_index : to_index])
        split_target.append(target[from_index : to_index])

    split_data = np.array(split_data)
    split_target = np.array(split_target)
    data = np.array(data)
    target = np.array(target)

    gnb = GaussianNB()

    print "Gaussain naive_bayes classifier selected..."

    # reverse the list
    # iris.target = iris.target[::-1]

    drift_dection = DriftDetiection()

    for i in xrange(len(split_data)):
        # print "Training model..."
        if i == 0:
            gnb.fit(split_data[i], split_target[i])

        print 'Predict...'
        # y_pred = gnb.predict(split_data)
        if i == (len(split_data) - 1):
            y_pred = gnb.predict(split_data[0])
            print "level: %s" % drift_dection.detect(y_pred, split_target[0], int(split_data[0].shape[0] / 2))
            print("Number of mislabeled points out of a total %d points : %d") % ((split_data[0].shape[0]), (split_target[0] != y_pred).sum())
        else:
            y_pred = gnb.predict(split_data[i + 1])
            print "level: %s" % drift_dection.detect(y_pred, split_target[i + 1], int(split_data[0].shape[0] / 2))
            print("Number of mislabeled points out of a total %d points : %d") % ((split_data[i + 1].shape[0]), (split_target[i + 1] != y_pred).sum())



        # words.map(lambda x: (x, 1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)
