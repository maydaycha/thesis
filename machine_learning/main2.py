# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.naive_bayes import GaussianNB
from pyspark.mllib.classification import NaiveBayes, LabeledPoint
from pyspark import SparkContext

from numpy import array

import arff
from sklearn.naive_bayes import GaussianNB
import math
import sys

levels = ['normal', 'warning', 'drift']

class DriftDetiection():

    def __init__(self):
        self.min_standard_deviation = 1
        self.min_error_probability = 1
        self.current_min = 1
        self.levels = ['normal', 'warning', 'drift']

    def detect(self, predict_labels, real_labels, k):
        print "K: %d" % k
        miss_label_count = (predict_labels != real_labels).sum()
        total_instance = len(predict_labels)
        print "total instance: %d" % total_instance
        precision_rate = (total_instance - miss_label_count) / total_instance
        miss_label_rate = miss_label_count / total_instance

        print "miss label: %f" % miss_label_rate

        # Binomial Distribution
        # 至少 k 個分錯的機率
        # P(Sn >= k) = 1 - ( P(Sn = 0) + P(Sn = 1)  + ..... + P(Sn = k-1) )
        error_probability = 0.0
        for i in range(k):
            #print "1: %f" % (math.factorial(total_instance) / (math.factorial(i) * math.factorial(total_instance - i)))
            #print "2: %f" % math.pow(miss_label_rate, i)
            #print "3：%f" % math.pow((1 - miss_label_rate), (total_instance - i))
            #print "3：%f, %f" % ((1 - miss_label_rate), (total_instance - i))
            error_probability += (math.factorial(total_instance) / (math.factorial(i) * math.factorial(total_instance - i))) * math.pow(miss_label_rate, i) * math.pow(1 - miss_label_rate, total_instance - i)

        error_probability = 1 - error_probability
        print "error probability: %f" % error_probability

        # standard deviation => s = math.sqrt(Pi * (1-Pi) / i)
        standard_deviation = math.sqrt(error_probability * (1 - error_probability) / total_instance)

        print "standard deviation: %f" % standard_deviation


        if (error_probability + standard_deviation) <= self.current_min:
            self.min_error_probability = error_probability
            self.min_standard_deviation = standard_deviation
            self.current_min = error_probability + standard_deviation
        print "current min: %f " % self.current_min
        print "min_error_probability: %f" % self.min_error_probability
        print "min_standard_deviation: %f" % self.min_standard_deviation

        # detect the level of concept drift
        if (error_probability + standard_deviation) >= (self.min_error_probability + 3 * self.min_standard_deviation):
            # if concept drift detected, then reset the state of these three variables
            self.current_min = 1
            self.min_standard_deviation = 1
            self.min_error_probability = 1
            return self.levels[2]
        elif (error_probability + standard_deviation) >= (self.min_error_probability + 2 * self.min_standard_deviation):
            return self.levels[1]
        else:
            return self.levels[0]



# round the float number absolutely
def ceiling(x):
    n = int(x)
    return n if n-1 < x <= n else n+1




if __name__ == '__main__':

    sc = SparkContext(appName="PythonNaiveBayes")

    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: main.py <split size>"
        exit(-1)

    split_size = int(sys.argv[1])

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

    split_trainingSet = []
    split_data = []
    split_targets = []

    concept_change_data = []

    for i in range(split_size):
        from_index = int(i * dataset_size / split_size)
        to_index = int((i+1) * dataset_size / split_size)
        print " %d : %d" % (from_index, to_index)

        split_data.append(data[from_index : to_index])
        split_targets.append(targets[from_index : to_index])


    # prepare LabeledPoint
    for ts, ds in zip(split_targets, split_data):
        split_trainingSet.append([LabeledPoint(t, a) for t, a in zip(ts, ds)])


    print "Gaussain naive_bayes classifier selected..."

    drift_dection = DriftDetiection()

    # covert list to numpy array
    split_targets = array(split_targets)

    for i, v in enumerate(split_trainingSet):
       # if i == 0:
        model = NaiveBayes.train(sc.parallelize(v))
        print '==================== round %d ==============================' % i

        print 'Predict...'

        if i == (len(split_trainingSet) - 1):
            y_pred = model.predict(sc.parallelize(split_data[0])).collect()
            y_pred = array(y_pred)

            print "Number of mislabeled points out of a total %d points : %d" % (len(split_data[0]), (split_targets[0] != y_pred).sum())

            level = drift_dection.detect(y_pred, split_targets[0], ceiling(len(split_data[0]) / 2))

            print "level: %s" % level

        else:
            print "@@ : %f" % len(split_data[i+1])
            y_pred = model.predict(sc.parallelize(split_data[i + 1])).collect()
            y_pred = array(y_pred)

            print "Number of mislabeled points out of a total %d points : %d" % (len(split_data[i + 1]), (split_targets[i + 1] != y_pred).sum())

            level = drift_dection.detect(y_pred, split_targets[i + 1], ceiling(len(split_data[i + 1]) / 2))

            print "level: %s" % level


        # words.map(lambda x: (x, 1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)
