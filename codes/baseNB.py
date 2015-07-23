from sklearn.naive_bayes import BaseNB
from sklearn import datasets
import numpy as np


if __name__ == '__main__':
    iris = datasets.load_iris()
    bnb = BaseNB()

    print "base naive_bayes classifier selected..."

    print "Training model and then predict....."

    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

    print("Number of mislabeled points out of a total %d points : %d") % ((iris.data.shape[0]), (iris.target != y_pred).sum())

