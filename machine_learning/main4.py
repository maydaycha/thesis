# -*- coding: utf-8 -*-

'''''

For skleanr Gaussian Naive Bayes

'''''

from __future__ import division
from pyspark.mllib.classification import NaiveBayes, LabeledPoint, NaiveBayesModel
from pyspark import SparkContext
from sklearn.naive_bayes import GaussianNB
from numpy import array
from myKNN import MyKNN

import arff, math, sys, csv, os

levels = ['normal', 'warning', 'drift']


class DriftDetiectionFramework():

    levels = ['normal', 'warning', 'drift']

    def __init__(self, outputFile = None):
        self.minStandardDeviation = 10
        self.minErrorProbability = 10
        self.currentMin = 10
        self.savedInstances = []
        self.outputFile = outputFile
        self.fieldnames = ["level", 'precision_rate', 'total_instance', 'miss_label_rate', 'error_probability', 'standard_deviation', "min_error_probability", 'self.minStandardDeviation', 'self.currentMin']
        self.csvWriter = None
        self.classifier = None
        self.retrainStrategy = False



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
           # print "1: %f" % (math.factorial(total_instance) // (math.factorial(i) * math.factorial(total_instance - i)))
           # print "2: %f" % math.pow(miss_label_rate, i)
           # print "3：%f" % math.pow((1 - miss_label_rate), (total_instance - i))
           # print "3：%f, %f" % ((1 - miss_label_rate), (total_instance - i))
            if total_instance < i:
                break

            error_probability += (math.factorial(total_instance) / (math.factorial(i) * math.factorial(total_instance - i))) * math.pow(miss_label_rate, i) * math.pow(1 - miss_label_rate, total_instance - i)

        error_probability = 1 - error_probability
        print "error probability: %f" % error_probability

        # standard deviation => s = math.sqrt(Pi * (1-Pi) / i)
        standard_deviation = math.sqrt(math.fabs(error_probability * (1 - error_probability) / total_instance))

        print "standard deviation: %f" % standard_deviation


        if (error_probability + standard_deviation) < self.currentMin and error_probability != 0 and standard_deviation != 0:
            self.minErrorProbability = error_probability
            self.minStandardDeviation = standard_deviation
            self.currentMin = error_probability + standard_deviation
        print "current min: %f " % self.currentMin
        print "min_error_probability: %f" % self.minErrorProbability
        print "min_standard_deviation: %f" % self.minStandardDeviation

        # Detect the level of concept drift
        if (error_probability + standard_deviation) > (self.minErrorProbability + 3 * self.minStandardDeviation):
            # If concept drift detected, then reset the state of these three variables
            self.currentMin = 10
            self.minStandardDeviation = 10
            self.minErrorProbability = 10
            l = levels[2]
        elif (error_probability + standard_deviation) > (self.minErrorProbability + 2 * self.minStandardDeviation):
            l = levels[1]
        else:
            l = levels[0]


        return {"level": l, 'precision_rate': precision_rate, 'total_instance': total_instance, 'miss_label_rate': miss_label_rate, 'error_probability': error_probability, 'standard_deviation': standard_deviation, "min_error_probability": self.minErrorProbability, 'self.minStandardDeviation': self.minStandardDeviation, 'self.currentMin': self.currentMin}


    def prepareClasasifier(self, trainingSet, partition = 0):
        features = [t[:-1] for t in trainingSet]
        labels = [t[-1] for t in trainingSet]
        self.classifier = GaussianNB().fit(array(features), array(labels))
        return self



    def predict(self, dataset, split_size):
        data = list(d[:-1] for d in dataset)
        targets = list((int(c[-1]) for c in dataset))

        # convert string to integer
        data = [map(float, x) for x in data]

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


        # Record the last level
        lastLevel = ''

        for i, v in enumerate(split_trainingSet):

            if self.retrainStrategy:
                trainset = self.getSavedInstance()
                if len(trainset) > 0:
                    fs = [t[:-1] for t in trainset]
                    ls = [t[-1] for t in trainset]
                    #self.classifier = GaussianNB().fit(array(fs), array(ls))
                self.retrainStrategy = False

            print '==================== round %d ==============================' % i

            print 'Predict...'

            if i == (len(split_trainingSet) - 1):
                predict_data = split_data[0]
                target_of_predict_data = split_targets[0]
            else:
                predict_data = split_data[i + 1]
                target_of_predict_data = split_targets[i + 1]

            predict_data = split_data[i]
            target_of_predict_data = split_targets[i]


            y_pred = self.classifier.predict(array(predict_data))

            print "Number of mislabeled points out of a total %d points : %d" % (len(predict_data), (target_of_predict_data != y_pred).sum())

            # Detect concept drift
            detection_result = self.detect(y_pred, target_of_predict_data, ceiling(len(predict_data) / 3))
           # detection_result = self.detect(y_pred, target_of_predict_data, 30)

            level = detection_result['level']

            print detection_result

            if level == 'normal' and lastLevel == 'warning':
                self.discardSaveInstance()

            if level == 'warning' or level == 'drift':
                self.saveInstance([x + [y] for x, y in zip(predict_data, target_of_predict_data)])

            if level == 'drift' and lastLevel != 'normal':
                self.retrainStrategy = True

            lastLevel = level

            print "level: %s" % level

            self.saveRecord(detection_result)



    def saveInstance(self, labeled_data):
        self.savedInstances.extend(labeled_data)
        return self


    def getSavedInstance(self):
        b = self.savedInstances[:]
        self.savedInstances = []
        print "clear saved isntances : %d " % len(self.savedInstances)
        return b

    def discardSaveInstance(self):
        self.savedInstances = []
        return self


    def saveRecord(self, record):
        if self.csvWriter == None:
            self.csvFile = open(self.outputFile, 'w')
            self.csvWriter = csv.DictWriter(self.csvFile, fieldnames = self.fieldnames)
            self.csvWriter.writeheader()
        else:
            self.csvWriter.writerow(record)
        return self


    def closeFile(self):
        if self.csvFile != None:
            self.csvFile.close()
        return self




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
            dataset = [s[:-1].split(',') for s in dataset]
            dataset = [map(lambda s : float(s), x) for x in dataset]

    #dataset = [d[:2] + [d[-1]] for d in dataset]

    #dataset = [map(int, d) for d in dataset]
    datasetLen = len(dataset)

    driftDectionFramework = DriftDetiectionFramework(outputFile)

    #driftDectionFramework.prepareClasasifier(dataset[:5000], -1 * int(5000 / 3))

    driftDectionFramework.prepareClasasifier(dataset[:100], -1 * int(len(dataset[:100])))

    partitionSize = 5000
    if datasetLen > partitionSize:
        for i in xrange(0, datasetLen, partitionSize):
            driftDectionFramework.predict(dataset[i : i + partitionSize], split_size)

    else:
        driftDectionFramework.predict(dataset, split_size)

    driftDectionFramework.closeFile()


