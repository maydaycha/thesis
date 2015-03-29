from pyspark import SparkContext
from sklearn.naive_bayes import MultinomialNB
from machine_learning.MyClassifier import MyClassifier
from pyspark.serializers import PickleSerializer

import numpy as np
import random

sc = SparkContext(appName="NoClass")


classifiers = []

data2 = [
    (1.0, [0.0, 0.0]),
    (0.0, [0.0, 1.0]),
    (1.0, [1.0, 0.0])
]

data3 = []

for i in range(100):
    data3.append(data2)

unLabeledData = [
    [0.0, 0.0],
    [1.0, 1.0],
    [1.0, 0.0]
]

serializer = PickleSerializer()

## init classifiers

#a = MultinomialNB()
#a.fit(np.array([[1.0, 0.0]]), np.array([1.0]))
#a.predict([1.0, 0.0])

for i in range(3):
    features = np.array([d[1] for d in data2])
    labels = np.array([d[0]for d in data2])
    classifiers.append(MultinomialNB().fit(features, labels))

#classifiers = sc.broadcast(classifiers)


bClf1 = sc.broadcast(classifiers[0])
bClf2 = sc.broadcast(classifiers[1])
bClf3 = sc.broadcast(classifiers[2])

#broadClassifiers = sc.broadcast(classifiers)

def predict(data):
    result = []
    #features = np.array([data[1]])
    #labels = np.array([data[0]])
    features = np.array([d[1] for d in data2])
    labels = np.array([d[0]for d in data2])

    feature = np.array(data[1])
    label = data[0]

    models = [bClf1.value, bClf2.value, bClf3.value]

    for c in models:
        try:
            predictLabel = c.predict(feature)
        except AttributeError:
            c = MultinomialNB().fit(features, labels)
            predictLabel = c.predict(feature)

        result.append((predictLabel, c))

    return (label, result)

def p2(data):
    for c in classifiers.value:
        label = c.predict([1.0, 0.0])
    print classifiers.value
    return (1, label)

def p3(data):
    return data[0]

r = sc.parallelize(data2).map(predict).reduce(lambda x,y: [x,y])

print r






