from pyspark import SparkContext
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

class MyClass(MultinomialNB):

    def __init__(self):
        MultinomialNB.__init__(self)

    def func(self, s):
        return s

    def doStuff(self, rdd):
        return rdd.map(self.func)


class My(NaiveBayes):
    def __init__(self):
        NaiveBayes.__init__(self)


if __name__ == '__main__':
    sc = SparkContext(appName="test")

    # an RDD of LabeledPoint
    data = sc.parallelize([
        LabeledPoint(0.0, [0.0, 0.0])
    ])

    # Train a naive Bayes model.
    model = My.train(data, 1.0)

    # Make prediction.
    prediction = model.predict([0.0, 0.0])

    print prediction
