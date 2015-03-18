from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np


if __name__ == '__main__':
    iris = datasets.load_iris()
    gnb = GaussianNB()

    print "Gaussain naive_bayes classifier selected..."

    # iris.target = iris.target[::-1]

    print "Training model and then predict....."

    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

    # print iris.data
    # print iris.data.shape

    print("Number of mislabeled points out of a total %d points : %d") % ((iris.data.shape[0]), (iris.target != y_pred).sum())

    # print gnb.predict(np.array([5, 3.3, 1.7, 2.5]))

    # print iris.target

    print "========"

    # print([(i, j) for i, j in enumerate(zip(iris.target, y_pred)) if i != j])
    # print zip(iris.target, y_pred)
    # for idx, (i, j) in enumerate(zip(iris.target, y_pred)):
        # if i != j:
            # print "%d %d %d data: %s" % (idx, i ,j, iris.data[idx])

    # print gnb.predict(np.array(iris.data[52]))
    # print iris.target[52]



