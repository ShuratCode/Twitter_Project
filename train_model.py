import csv
import re
import string

from nltk.corpus import stopwords


class TrainModel:

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

    def read_csv(self):

        db_csv = open(self.__DB_path, 'r', encoding='utf8', errors='ignore')

        reader = csv.reader(self.db_csv, delimiter=',')
        headers = next(reader)
        text_idx = headers.index('text')
        gender_idx = headers.index('gender')
        terms_male, terms_female, terms_brand = self.build_terms(text_idx, gender_idx, reader)

        # TODO: train models on this data.

    def build_terms(self, text_idx, gender_idx, reader):

        """
        Build lists of terms used by male/female/brand
        :param text_idx: The text column
        :param gender_idx: The gender column
        :param reader:
        :return:
        """
        terms_male = []
        terms_female = []
        terms_brand = []
        punctuation = list(string.punctuation)
        stop_words = stopwords.words('english') + punctuation + ['rt', 'via']

        for line in reader:
            tokens = self.preprocess(line[text_idx])
            for token in tokens:
                if token not in stop_words:
                    if line[gender_idx] == 'male':
                        terms_male.append(token)
                    if line[gender_idx] == 'female':
                        terms_female.append(token)
                    if line[gender_idx] == 'brand':
                        terms_brand.append(token)

        return terms_male, terms_female, terms_brand

    def tokenize(self):
        s = re.sub(r'[^\x00-\x7f]*', r'', s)
        return self.__tokens_re.findall(s)

    def preprocess(self, lowercase=False):
        tokens = self.tokenize()
        if lowercase:
            tokens = [token if self.__emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens
