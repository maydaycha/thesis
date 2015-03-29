from sklearn.naive_bayes import MultinomialNB


class MyClassifier(MultinomialNB):

    def __init__(self):
        MultinomialNB.__init__(self)
        self.weight = 0
