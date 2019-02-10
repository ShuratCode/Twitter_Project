import re
import string
from collections import Counter
import pprint

import pandas as pd
from nltk.corpus import stopwords


class DataPresentation:

    def __init__(self):
        self.__DB_path = 'gender-classifier-DFE-791531.csv'

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
        self.data_frame = None

    def read_csv(self):
        """
        Read the csv file and present it
        :return: representation of the
        """
        data = pd.read_csv(self.__DB_path, encoding='latin-1')
        pd.value_counts(data['gender']).plot.bar()
        return data

    def build_terms(self, data_frame):

        """
        Build lists of terms used by male/female/brand
        :param: data_frame data frame that holds the text and gender fields of the DB
        :return:
        """
        terms_male = []
        terms_female = []
        terms_brand = []
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
                    if line['gender'] == 'brand':
                        terms_brand.append(token)
        return terms_male, terms_female, terms_brand

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

    def data_to_df(self, data):
        """
        convert the data to data frame and clean it up
        :param data: data read from csv file
        :return: clean data frame
        """
        self.data_frame = data[['text', 'gender']]
        self.data_frame.dropna(inplace=True)
        self.data_frame = self.data_frame[self.data_frame.gender != 'unknown']

    def print_common(self):
        terms_male, terms_female, terms_brand = self.build_terms(self.data_frame)

        pp = pprint.PrettyPrinter()

        count_male = Counter()
        count_male.update(terms_male)
        print('Male most common terms: ', sum(count_male.values()))
        pp.pprint(count_male.most_common(20))
        print('=' * 80)

        count_female = Counter()
        count_female.update(terms_female)
        print('Female most common terms: ', sum(count_female.values()))
        pp.pprint(count_female.most_common(20))
        print('=' * 80)

        count_brand = Counter()
        count_brand.update(terms_brand)
        print('Brand most common terms: ', sum(count_brand.values()))
        pp.pprint(count_brand.most_common(20))
