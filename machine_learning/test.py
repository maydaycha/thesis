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


def myFunc(data):
    import random
    with open("/home/maydaycha/sea.data") as f:
        content = f.readlines()


    #with open("/home/maydaycha/myresult.txt", 'w') as f:
     #   for c in content:
      #      f.write("%s" % c)

    return[content]

if __name__ == '__main__':
    sc = SparkContext(appName="test")

    myRDD = sc.parallelize(range(6), 3)
    r = sc.runJob(myRDD, myFunc)

    with open('/home/maydaycha/myresult.txt', 'w') as f:
        for c in r:
            for x in c:
                f.write("%s" % x)
