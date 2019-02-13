import operator
from time import time

import numpy
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


class TrainModels:
    def __init__(self, data_frame):
        self.train, self.test = train_test_split(data_frame, test_size=0.2)
        self.__form_to_text()
        self.__clfs = {"SVM": SGDClassifier(), "Perceptron": Perceptron()}
        self.X_train, self.X_test = self.__tf_idf_feature_extraction()
        self.Y_train, self.Y_test = self.train.gender, self.test.gender
        numpy.random.seed(7)

    def __tf_idf_feature_extraction(self):
        """
        Create features from the data set based on tf_idf method
        :return: list for features in the training set and data set
        """
        print('=' * 80)
        print("TF-IDF Feature Extraction")
        t0 = time()
        vectorizer = TfidfVectorizer()
        vec_train = vectorizer.fit_transform(self.train.text)
        vec_test = vectorizer.transform(self.test.text)
        duration = time() - t0
        print("DONE!!!!! total time: %fs" % duration)
        print('=' * 80)
        return vec_train, vec_test

    def __form_to_text(self):
        """
        transform the text field in the training set to be str
        :return:
        """
        for _, row in self.train.iterrows():
            self.train['text'] = str(self.train['text'])

    def __benchmark(self, clf):
        """
        Training clf model on the training set
        :param clf: the model to train
        :return: the accuracy score of the training
        """
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

        return score

    def train_models(self):
        """
        Train the model and compare them
        :return: the best model
        """
        scores = {}
        for name, clf in self.__clfs.items():
            print('=' * 80)
            print(name)
            scores[name] = self.__benchmark(clf)
        best = self.__get_best_score(scores)
        best_params = self.optimize(self.__clfs[best])
        simple_score, simple_model = self.__run_best_model(best_params, best)
        keras_model, keras_score = self.__train_sequential()
        if keras_score >= simple_score:
            return keras_model
        else:
            return simple_model

    def __run_best_model(self, best_params, best_clf):
        """
        run the best model after optimization
        :param best_params: the best parameters the optimization selected
        :param best_clf: the best simple model name the optimizer chose
        :return: the score of the model and the model itself
        """
        tf_idf = TfidfVectorizer(stop_words=best_params.get('vect__stop_words'))
        print('=' * 80)
        print("TF-IDF feature extraction")
        t0 = time()
        train = tf_idf.fit_transform(self.train.text)
        test = tf_idf.transform(self.test.text)
        train_time = time() - t0
        print("DONE!!! total time: %fs" % train_time)
        print('=' * 80)
        clf_func = self.__clfs[best_clf]
        print('Training:')
        t0_training = time()
        clf_func.fit(train, self.train.gender)
        train_time = time() - t0_training
        print("train time: %0.3fs" % train_time)
        t_test = time()
        pred = clf_func.predict(test)
        test_time = time() - t_test
        print("test time: %0.3fs" % test_time)
        score = metrics.accuracy_score(self.test.gender, pred)
        print('accuracy:  ' + str(score))
        return score, clf_func

    @staticmethod
    def __get_best_score(scores):
        """
        get the best clf score
        :param scores: the score dictionary
        :return: the name of the best clf
        """
        best = max(scores.items(), key=operator.itemgetter(1))[0]
        print("The best classification for this corpus is: " + str(best))
        return best

    def optimize(self, best_func):
        """
        Optimize the model by using grid search
        :param best_func: the function of the best model
        :return: the best params for the model
        """
        nb_clf = Pipeline(steps=[('vect', TfidfVectorizer()), ('clf', best_func)])
        parameters = {
            'vect__stop_words': [None, 'english'],
        }
        gs_clf = GridSearchCV(nb_clf, parameters, scoring='accuracy')
        gs_clf = gs_clf.fit(self.train.text, self.train.gender)
        print("Best parameters: " + str(gs_clf.best_params_))
        print('Best score: ' + str(gs_clf.best_score_))
        print('=' * 80)
        return gs_clf.best_params_

    def __train_sequential(self):
        """
        Train the Sequential model with embedding dens and LSTM layers
        :return: the model and its score
        """
        x_train = numpy.array(self.train.text)
        y_train = numpy.array(self.train.gender)
        x_test = numpy.array(self.test.text)
        y_test = numpy.array(self.test.gender)
        # keras_X_train = sequence.pad_sequences(self.train.text, maxlen=max_text_length)
        # keras_X_test = sequence.pad_sequences(self.test.text, maxlen=max_text_length)
        model = Sequential()
        model.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=1))
        model.add(Dense(output_dim=6, init='uniform', activation='relu'))
        model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)
        keras_score = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (keras_score[1] * 100))
        return model, keras_score[1]
