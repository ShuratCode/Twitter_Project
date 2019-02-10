import csv
import re
import string
import pandas as pd

from nltk.corpus import stopwords


class DataPresentation:

    def __init__(self):
        self.__DB_path = './gender-classifier-DFE-791531.csv'

        self.__emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""

        self.__regex_str = [
            self.__emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]

        self.__tokens_re = re.compile(r'(' + '|'.join(self.__regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        self.__emoticon_re = re.compile(r'^' + self.__emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
        data_frame = self.read_csv()
        terms_male, terms_female = self.build_terms(data_frame)


    def read_csv(self):
        """
        Read the csv file and transform it to data frame. Will save only the text and gender fields and clean up a little
        :return: data frame that represent the csv file
        """

        data = pd.read_csv(self.__DB_path, encoding='utf8')
        data_frame = data[['text', 'gender']]
        data_frame = data_frame.dropna(inplace=True)
        data_frame = data_frame[data_frame.gender != 'unknown']
        return data_frame


    def build_terms(self, data_frame):

        """
        Build lists of terms used by male/female/brand
        :param: data_frame data frame that holds the text and gender fields of the DB
        :return:
        """
        terms_male = []
        terms_female = []
        punctuation = list(string.punctuation)
        stop_words = stopwords.words('english') + punctuation + ['rt', 'via']

        for index, line in data_frame.iterrows():
            tokens = self.preprocess(line['text'], lowercase=True)
            for token in tokens:
                if token not in stop_words:
                    if line['gender'] == 'male':
                        terms_male.append(token)
                    if line['gender'] == 'female':
                        terms_female.append(token)
        return terms_male, terms_female

    def tokenize(self, tweet_text):
        """
        Tokenize the text of a tweet
        :param tweet_text: text filed of the tweet
        :return: list of tokens
        """
        tweet_text = re.sub(r'[^\x00-\x7f]*', r'', tweet_text)
        return self.__tokens_re.findall(tweet_text)

    def preprocess(self, tweet_text, lowercase=False):
        tokens = self.tokenize(tweet_text)
        if lowercase:
            tokens = [token if self.__emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens
