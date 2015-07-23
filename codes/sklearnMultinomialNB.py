from __future__ import division
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import numpy as np
import csv

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]



filename = 'myData.csv'
# filename = 'sea.data'
with open(filename, 'rU') as f:
    reader = csv.reader(f)
    content = list(reader)

content = [map(float, z) for z in content]
features = np.array([d[:-1] for d in content])
labels = np.array([d[-1] for d in content])

clf = GaussianNB()

clf.fit(features[:10], labels[:10])

# clf.partial_fit(features[:10], labels[:10], np.array([0, 1]))

print ""

y_pred = []
r_labels = []
n = 10
print len(features)
for i in xrange(0, len(features), n):
    if i % 100 == 0:
        print "================"

    y_pred = clf.predict(features[i:i+n])
    r_labels = labels[i:i+n]
    print "precisous reate: %f" % clf.score(features[i:i+n], r_labels)


    # break
    # clf.partial_fit(features[i:i+n], labels[i:i+n], None)

# for idx, feature in enumerate(features):
    # y_pred.append(clf.predict(feature)[0])

# print y_pred

print np.array(r_labels).shape
print np.array(y_pred).shape

# with open('my2.csv', 'w') as f:
    # for i in range(len(y_pred)):
        # f.write("%f\n" % ((y_pred[i] == r_labels[i]).sum() / len(r_labels[i])) )

# print (labels != np.array(y_pred)).sum()


# print clf.score(features, labels)

# print clf.class_prior_

print ""
# print clf.theta_
print ""
# print clf.sigma_
print ""
# print clf.predict_proba(features[-5])
