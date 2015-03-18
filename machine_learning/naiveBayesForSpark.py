from pyspark.mllib.classification import NaiveBayes, LabeledPoint
from pyspark import SparkContext

sc = SparkContext(appName="PythonNaiveBayes")

data = [
    LabeledPoint(0.0, [0.0, 0.0]),
    LabeledPoint(0.0, [0.0, 1.0]),
    LabeledPoint(1.0, [1.0, 0.0])
]

model = NaiveBayes.train(sc.parallelize(data))
model.predict(array([0.0, 1.0]))
