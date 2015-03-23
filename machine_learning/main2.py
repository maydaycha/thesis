# -*- coding: utf-8 -*-

from __future__ import division
from pyspark.mllib.classification import NaiveBayes, LabeledPoint
from pyspark import SparkContext
from numpy import array

import arff
import math
import sys
import csv
import os

levels = ['normal', 'warning', 'drift']

class DriftDetiectionFramework():

    def __init__(self, sparkContext, inputFile = None, outputFile = None):
        self.min_standard_deviation = 1
        self.min_error_probability = 1
        self.current_min = 1
        self.levels = ['normal', 'warning', 'drift']
        self.saved_instances = []
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.fieldnames = ["level", 'precision_rate', 'miss_label_rate', 'error_probability', 'standard_deviation', "min_error_probability", 'self.min_standard_deviation', 'self.current_min']
        self.sc = sparkContext

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
            print "1: %f" % (math.factorial(total_instance) // (math.factorial(i) * math.factorial(total_instance - i)))
            print "2: %f" % math.pow(miss_label_rate, i)
            print "3：%f" % math.pow((1 - miss_label_rate), (total_instance - i))
            print "3：%f, %f" % ((1 - miss_label_rate), (total_instance - i))
            error_probability += (math.factorial(total_instance) // (math.factorial(i) * math.factorial(total_instance - i))) * math.pow(miss_label_rate, i) * math.pow(1 - miss_label_rate, total_instance - i)

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
            l = self.levels[2]
        elif (error_probability + standard_deviation) >= (self.min_error_probability + 2 * self.min_standard_deviation):
            l = self.levels[1]
        else:
            l = self.levels[0]


        return {"level": l, 'precision_rate': precision_rate, 'miss_label_rate': miss_label_rate, 'error_probability': error_probability, 'standard_deviation': standard_deviation, "min_error_probability": self.min_error_probability, 'self.min_standard_deviation': self.min_standard_deviation, 'self.current_min': self.current_min}


    def predict(self, dataset, split_size):
        data = list(d[:-1] for d in dataset)
        targets = list((int(c[-1]) for c in dataset))

        # convert string to integer
        data = [map(int, x) for x in data]

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


            # covert list to numpy array
        split_targets = array(split_targets)

        retrain_strategy = False

        for i, v in enumerate(split_trainingSet):

            if i == 0:
                model = NaiveBayes.train(self.sc.parallelize(v))
            elif retrain_strategy == True:
                model = NaiveBayes.train(self.sc.parallelize(getSavedInstance()))
                retrain_strategy = False

            print '==================== round %d ==============================' % i

            print 'Predict...'

            if i == (len(split_trainingSet) - 1):
                predict_data = split_data[0]
                target_of_predict_data = split_targets[0]
            else:
                predict_data = split_data[i + 1]
                target_of_predict_data = split_targets[i + 1]

            y_pred = model.predict(self.sc.parallelize(predict_data)).collect()
            y_pred = array(y_pred)

            print "Number of mislabeled points out of a total %d points : %d" % (len(predict_data), (target_of_predict_data != y_pred).sum())

            detection_result = detect(y_pred, target_of_predict_data, ceiling(len(predict_data) / 2))

            level = detection_result['level']

            print detection_result

            if level == 'warning' or level == 'drift':
                saveInstance([LabeledPoint(x, y) for x, y in zip(target_of_predict_data, predict_data)])
                retrain_strategy = True

            print "level: %s" % level

            # writer.writerow(detection_result)
            saveRecord(detection_result)


            # words.map(lambda x: (x, 1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)

        closeFile()





    def saveInstance(self, labeled_data):
        self.saved_instances.extend(labeled_data)



    def getSavedInstance(self):
        b = self.saved_instances[:]
        self.saved_instances = []
        print "clear saved isntances : %d " % len(self.saved_instances)
        return b

    def saveRecord(self, record):
        if self.csvWriter == None:
            self.csvFile = open(self.outputFile, 'w')
            self.csvWriter = csv.DictWriter(self.csvFile, fieldnames = self.fieldnames)
            self.csvWriter.writeheader()
        else:
            self.csvWriter.writerow(record)

    def closeFile(self)
        if self.csvFile != None:
            self.csvFile.close()




# round the float number absolutely
def ceiling(x):
    n = int(x)
    return n if n-1 < x <= n else n+1


if __name__ == '__main__':
    sc = SparkContext(appName="NaiveBayes")

    if len(sys.argv) < 4:
        print >> sys.stderr, "Usage: main.py <split size> <input filepath> <output outputFile>"
        exit(-1)

    split_size = int(sys.argv[1])
    inputFile = sys.argv[2]
    outputFile = sys.argv[3]

    # load dataset
    # file_path = 'hdfs://ubuntu-iim:9000/concept_drift_data/usenet1.arff'
    # file = sc.textFile(file_path)
    name, extension = os.path.splitext(inputFile)
    if extension == '.arff':
        dataset = arff.load(open(inputFile, 'rb'))
        dataset = dataset['data']
    else:
        with open(inputFile) as f:
            dataset = f.readlines()
            dataset = dataset[:5000]
            dataset = [s[:-1].split(',') for s in dataset]
            dataset = [map(lambda s : float(s), x) for x in dataset]

    driftDectionFramework = DriftDetiectionFramework(sc, inputFile, outputFile)

    driftDectionFramework.predict(dataset, split_size)





# if __name__ == '__main__':

#     sc = SparkContext(appName="NaiveBayes")

#     fieldnames = ["level", 'precision_rate', 'miss_label_rate', 'error_probability', 'standard_deviation', "min_error_probability", 'self.min_standard_deviation', 'self.current_min']


#     if len(sys.argv) < 4:
#         print >> sys.stderr, "Usage: main.py <split size> <input filepath> <output outputFile>"
#         exit(-1)

#     split_size = int(sys.argv[1])
#     inputFile = sys.argv[2]
#     outputFile = sys.argv[3]

#     # load dataset
#     # file_path = 'hdfs://ubuntu-iim:9000/concept_drift_data/usenet1.arff'
#     # file = sc.textFile(file_path)
#     name, extension = os.path.splitext(inputFile)
#     if extension == '.arff':
#         dataset = arff.load(open(inputFile, 'rb'))
#         dataset = dataset['data']
#     else:
#         with open(inputFile) as f:
#             dataset = f.readlines()
#             dataset = dataset[:5000]
#             dataset = [s[:-1].split(',') for s in dataset]
#             dataset = [map(lambda s : float(s), x) for x in dataset]

#     print "load input file"

#     data = list(d[:-1] for d in dataset)
#     targets = list((int(c[-1]) for c in dataset))

#     # convert string to integer
#     data = [map(int, x) for x in data]

#     dataset_size = len(data)

#     split_trainingSet = []
#     split_data = []
#     split_targets = []

#     concept_change_data = []

#     for i in range(split_size):
#         from_index = int(i * dataset_size / split_size)
#         to_index = int((i+1) * dataset_size / split_size)
#         print " %d : %d" % (from_index, to_index)

#         split_data.append(data[from_index : to_index])
#         split_targets.append(targets[from_index : to_index])


#     # prepare LabeledPoint
#     for ts, ds in zip(split_targets, split_data):
#         split_trainingSet.append([LabeledPoint(t, a) for t, a in zip(ts, ds)])


#     print "Gaussain naive_bayes classifier selected..."

#     drift_dection = DriftDetiection()

#     # covert list to numpy array
#     split_targets = array(split_targets)

#     csvfile = open(outputFile, 'w')
#     writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
#     writer.writeheader()

#     retrain_strategy = False

#     for i, v in enumerate(split_trainingSet):

#         if i == 0:
#             model = NaiveBayes.train(sc.parallelize(v))
#         elif retrain_strategy == True:
#             model = NaiveBayes.train(sc.parallelize(drift_dection.getSavedInstance()))
#             retrain_strategy = False

#         print '==================== round %d ==============================' % i

#         print 'Predict...'

#         if i == (len(split_trainingSet) - 1):
#             predict_data = split_data[0]
#             target_of_predict_data = split_targets[0]
#         else:
#             predict_data = split_data[i + 1]
#             target_of_predict_data = split_targets[i + 1]

#         y_pred = model.predict(sc.parallelize(predict_data)).collect()
#         y_pred = array(y_pred)

#         print "Number of mislabeled points out of a total %d points : %d" % (len(predict_data), (target_of_predict_data != y_pred).sum())

#         detection_result = drift_dection.detect(y_pred, target_of_predict_data, ceiling(len(predict_data) / 2))

#         level = detection_result['level']

#         print detection_result

#         if level == 'warning' or level == 'drift':
#             drift_dection.saveInstance([LabeledPoint(x, y) for x, y in zip(target_of_predict_data, predict_data)])
#             retrain_strategy = True

#         print "level: %s" % level

#         writer.writerow(detection_result)


#         # words.map(lambda x: (x, 1)).reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)

#     csvfile.close()
