

if __name__ == '__main__':
    from sklearn import svm, datasets
    from sklearn.externals import joblib
    import numpy as np
    import sys
    import os.path



    if len(sys.argv) < 2:
        print "use iris data"
    else:
        print "SVM model start, use %s as training data sets, %s as target data sets" % (sys.argv[1], sys.argv[2])
        training_sets = []
        target_sets = []

        with open(sys.argv[1]) as t1, open(sys.argv[2]) as t2:
            training_sets = t1.read()
            target_sets = t2.read()


        training_sets = np.array(eval(training_sets))
        target_sets = np.array(eval(target_sets))


        # if model exists, use it
        model_filename = 'svm.pkl'

        if os.path.isfile(model_filename):
            print "model exists, loading..."
            clf = joblib.load(model_filename)
        else:
            print "training model..."
            clf = svm.SVC()
            # traning model
            clf.fit(training_sets[:-1 * len(training_sets) / 3 ], target_sets[:-1 * len(target_sets) / 3])


        # prediction
        print "start predicting..."
        y_pred = clf.predict(training_sets[-1 * 1/3 * len(training_sets):])

        print("Number of mislabeled points out of a total %d points : %d") % ((training_sets.shape[0]), (target_sets != y_pred).sum())

        with open('result.txt', 'w') as f:
            f.write("%s" % y_pred)

        # persistence model

        from sklearn.externals import joblib
        joblib.dump(clf, 'svm.pkl')

        # print training_sets
        # print target_sets







