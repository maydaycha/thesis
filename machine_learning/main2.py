# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.naive_bayes import GaussianNB
from pyspark.mllib.classification import NaiveBayes, LabeledPoint
from pyspark import SparkContext

import numpy as np
import arff
from sklearn.naive_bayes import GaussianNB
import math
import sys

levels = ['normal', 'warning', 'drift']

class DriftDetiection():

    def __init__(self):
        self.min_standard_deviation = 1
        self.min_error_rate = 1
        self.current_min = 1
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

        error_rate = 1 - error_rate
        print "error rate: %f" % error_rate

        # standard deviation => s = math.sqrt(Pi * (1-Pi) / i)
        standard_deviation = math.sqrt(error_rate * (1 - error_rate) / total_instance)

        print "standard deviation: %f" % standard_deviation


        if (error_rate + standard_deviation) <= self.current_min:
            self.min_error_rate = error_rate
            self.min_standard_deviation = standard_deviation
            self.current_min = error_rate + standard_deviation
        print "current min: %f " % self.current_min
        print "min_error_rate: %f" % self.min_error_rate
        print "min_standard_deviation: %f" % self.min_standard_deviation

        # detect the level of concept drift
        if (error_rate + standard_deviation) >= (self.min_error_rate + 3 * self.min_standard_deviation):
            self.current_min = 1
            self.min_standard_deviation = 1
            self.min_error_rate = 1
            return self.levels[2]
        elif (error_rate + standard_deviation) >= (self.min_error_rate + 2 * self.min_standard_deviation):
            return self.levels[1]
        else:
            return self.levels[0]







if __name__ == '__main__':
    sc = SparkContext(appName="PythonNaiveBayes")

    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: main.py <split size>"
        exit(-1)

    # load dataset
    # file_path = 'hdfs://ubuntu-iim:9000/concept_drift_data/usenet1.arff'
    # file = sc.textFile(file_path)
    dataset = arff.load(open('/home/maydaycha/spark/usenet1.arff', 'rb'))

    print "load arff file"

    data = list(d[:-1] for d in dataset['data'])

    # convert string to integer
    data = [map(int, x) for x in data]

    targets = list((int(c[-1]) for c in dataset['data']))

    dataset_size = len(data)

    split_size = int(sys.argv[1])

    split_trainingSet = []
    split_data = []
    split_targets = []

    for i in range(split_size):
        from_index = int(i * dataset_size / split_size)
        to_index = int((i+1) * dataset_size / split_size)
        print " %d : %d" % (from_index, to_index)
        #split_data.append(split[from_index : to_index])
        #split_target.append(target[from_index : to_index])

        split_data.append(data[from_index : to_index])
        split_targets.append(targets[from_index : to_index])


    # prepare LabeledPoint
    for ts, ds in zip(split_targets, split_data):
        split_trainingSet.append([LabeledPoint(t, a) for t, a in zip(ts, ds)])


    print "============="
    print np.array(split_targets).shape
    print np.array(split_trainingSet).shape


    print "Gaussain naive_bayes classifier selected..."

    # reverse the list
    # iris.target = iris.target[::-1]

    drift_dection = DriftDetiection()


    for i, v in enumerate(split_trainingSet):
        # print "Training model..."
       # if i == 0:
        model = NaiveBayes.train(sc.parallelize(v))
        print '==================== round %d ==============================' % i

        print 'Predict...'

        if i == (len(split_trainingSet) - 1):
            y_pred = model.predict(sc.parallelize(split_data[0])).collect()

            print "level: %s" % drift_dection.detect(np.array(y_pred), np.array(split_targets[0]), int(len(split_data[0]) / 2))
            print "Number of mislabeled points out of a total %d points : %d" % (len(split_data[0]), (np.array(split_targets[0]) != np.array(y_pred)).sum())
        else:
            y_pred = model.predict(sc.parallelize(split_data[i + 1])).collect()
            print "level: %s" % drift_dection.detect(np.array(y_pred), np.array(split_targets[i + 1]), int(len(split_data[i + 1]) / 2))
            print "Number of mislabeled points out of a total %d points : %d" % (len(split_data[i + 1]), (np.array(split_targets[i + 1]) != np.array(y_pred)).sum())



        # words.map(lambda x: (x, 1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)