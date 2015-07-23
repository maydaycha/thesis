from pyspark.mllib.classification import NaiveBayes, LabeledPoint
from pyspark import SparkContext
import sys
import numpy as np
import random
from operator import add
from sklearn.naive_bayes import MultinomialNB

sc = SparkContext(appName="PythonNaiveBayes")

#model = NaiveBayes.train(sc.parallelize(data))
#model.predict(array([0.0, 1.0]))


class MyClass():
    def __init__(self):
        self.models = []

    def prepareModels(self, dataset = None, numberOfModel = 0):
        if numberOfModel == 0 or dataset == None:
            print >> sys.stderr, "No model or dataset!"
            exit(-1)

        for i in range(numberOfModel):
            self.models.append(NaiveBayes.train(sc.parallelize(dataset)))

        return self


    def predict(self, unLabeledData):
        parallelizeData = sc.parallelize(unLabeledData)
        yPreds = []
        for model in self.models:
            yPred = model.predict(parallelizeData)
            yPreds.append(yPred)

        return yPreds

    def predict2(self, unLabeledData):
        def tt(x):
            return (1,x)

        def parallelPredict(unLabeledData):
            #for idx, model in enumerate(self.models):
           # yield (idx, self.models[0].predict(unLabeledData))
           return (1, NaiveBayes.train(sc.parallelize(dataset)).predict(unLabeledData))

       #return sc.parallelize(unLabeledData).map(parallelPredict).reduceByKey(lambda x, y: [x, y]).collect()
   # return self.sc.parallelize(unLabeledData).map(lambda x : ('1', x)).reduceByKey(lambda x, y : [x,y]).collect()


    def parallelPredict(self, unlabeledData):
        return (1, 1)
    #for idx, model in enumerate(self.models):
         #   yield {idx: idx}
           # yield (idx, model.predict(unlabeledData))




def test(x):
    return (1, x)


# Now using !!!
class MyClass2():

    beta = 0.5
    period = 1  # period that will update weigiht of classifier

    def __init__(self):
        self.classifiers = []
        self.predictErrorCount = 0
        self.predictCorrectCount = 0


    def prepareModels(self, dataset = None, numberOfModel = 0):
        if numberOfModel == 0 or dataset == None:
            print >> sys.stderr, "No model or dataset!"
            exit(-1)

        size = len(dataset)

        fromIndex = random.randint(0, size)

        for i in range(numberOfModel):
            #X = np.random.randint(5, size=(6, 100))
            #y = np.array([1, 2, 3, 4, 5, 6])
            X = np.array([d[1] for d in dataset])
            y = np.array([d[0] for d in dataset])
            clf = Classifier()
            clf.fit(X, y)
            self.classifiers.append(clf)

        return self

    def predict(self, data):

        # predict stage
        result = []
        label = data[0]
        features = data[1]
        predictScorePair = []

        self.classifiers = sc.broadcast(self.classifiers)

        for classifier in self.classifiers.value:
            predictLabel = classifier.predict(features)
            if predictLabel != label:
                classifier.setWeight(classifier.getWeight() * beta)
                classifier.predictErrorCount += 1
            else:
                classifier.predictCorrectCount += 1

            predictScorePair.append((predictLabel, classifier.getWeight()))

        return (label, predictScorePair)




class Classifier(MultinomialNB):

    def __init__(self):
        MultinomialNB.__init__(self)
        self.weight = 0
        self.positiveScore = 0
        self.negativeScore = 0
        self.predictErrorCount = 0
        self.predictCorrectCount = 0

    def setWeight(self, weight):
        self.weight = weight
        return self

    def getWeight(self):
        return self.weight


def predict(data):

    # predict stage
    result = []
    label = data[0]
    features = data[1]
    predictScorePair = []

    for classifier in classifiers:
        predictLabel = classifier.predict(features)
        if predictLabel != label:
            classifier.setWeight(classifier.getWeight() * beta)
            classifier.predictErrorCount += 1
        else:
            classifier.predictCorrectCount += 1

        predictScorePair.append((predictLabel, classifier.getWeight()))

    return (label, predictScorePair)


if __name__ == '__main__':

    data = [
            LabeledPoint(0.0, [0.0, 0.0]),
            LabeledPoint(0.0, [0.0, 1.0]),
            LabeledPoint(1.0, [1.0, 0.0])
            ]

    data2 = [
            (0.0, [0.0, 0.0]),
            (0.0, [0.0, 1.0]),
            (1.0, [1.0, 0.0])
            ]

    unLabeledData = [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0]
            ]


    myClass2 = MyClass2()



    rdd = sc.parallelize(data2).map(myClass2.predict).reduce(lambda x, y: [x, y])

    print rdd


    exit(0)



    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = MultinomialNB()
    clf.fit(X, y)
    print(clf.predict(X[2]))

    def t(x):
        X = np.random.randint(5, size=(6, 100))
        y = np.array([1, 2, 3, 4, 5, 6])
        clf = MultinomialNB()
        clf.fit(X, y)
        ran = random.randint(0, 5)
        result = clf.predict(X[ran])
        return (ran, result)


    result = sc.parallelize(X).map(t).reduceByKey(lambda x,y: [x,y]).collect()
    print result


