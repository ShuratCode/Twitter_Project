from time import time

from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


class TrainModels:
    def __init__(self, data_frame):
        self.train, self.test = train_test_split(data_frame, test_size=0.2)
        self.form_to_text()
        self.__clfs = ((SGDClassifier(), "SVM"), (Perceptron(), "Perceptron"), (MultinomialNB(), "Naive Bayes"))
        self.X_train = None
        self.X_test = None
        self.Y_train, self.Y_test = self.train.gender, self.test.gender

    def form_to_text(self):
        for _, row in self.train.iterrows():
            self.train['text'] = str(self.train['text'])

    def benchmark(self, clf):
        print('=' * 80)
        print('Training: ')
        print(clf)
        train_start = time()
        clf.fit(self.X_train, self.Y_train)
        train_time = time() - train_start
        print("The training time was: %0.3fs" % train_time)

        test_start = time()
        pred = clf.predict(self.X_test)
        test_time = time() - test_start
        print("The test time was: %0.3fs" % test_time)

        score = metrics.accuracy_score(self.Y_test, pred)
        print("accuracy: %0.3f" % score)

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
