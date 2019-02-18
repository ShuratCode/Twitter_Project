import re
import string
from collections import Counter
import pprint
import tweepy
import json
from nltk.stem import PorterStemmer
from keras_preprocessing.text import Tokenizer


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
        self.ps = PorterStemmer()

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
                    token = self.ps.stem(token)
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
        """
        Pre-process the text and tokens.
        :param tweet_text: the text to process
        :param lowercase: boolean, true if wanted to change all text to lowercase, false otherwise
        :return: list of clean tokens
        """
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
        """
        Working on the data frame to create list of terms for male, female and brand.
        Print the top 20 of each list
        """
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

    @staticmethod
    def find_most_common_country(data):

        countries = data['tweet_location'].value_counts()
        top_countries = []
        for loc, num in countries.items():
            top_countries.append((loc, num))

        return top_countries[0:6]

    @staticmethod
    def collect_tweets(file_location):

        api_key = "GXRCrOUdw2q35n8DduLSWYCGL"
        api_secret_key = "YnI09Fgy07LqPVyHPkx0NlodNoqyEuP3JvCSdlR1vKseszxsM3"

        access_token = "734660944920465408-ORnK69HUaAKirszN1pW12XfzBGyghrB"
        access_token_secret = "WrgnSATTGPVP7yJVQvCmC0vqZtGUHAtzTgqRcmQ3Uhvyo"

        authenticator = tweepy.OAuthHandler(api_key, api_secret_key)
        authenticator.set_access_token(access_token, access_token_secret)
        twitter_api = tweepy.API(authenticator)

        stream_listener = TwitterStreamer(file_location, twitter_api)
        stream_listener = tweepy.Stream(auth=twitter_api.auth, listener=TwitterStreamer)
        # stream_listener.filter(locations=[-0.510375, 51.28676, 0.334015, 51.691874])
        stream_listener.filter(track=['python'], is_async=False)

    def process_tweets(self, file_location):
        """
        Processes the tweets in the same manner as in Question 1
        :param file_location: Location of the tweets json
        """
        tweet_data = []
        tweets_json = open(file_location, 'r')

        for line in tweets_json:
            try:
                curr_tweet = json.loads(line)
                tweet_data.append(curr_tweet['text'])
            except Exception as ex:
                print(ex)
                continue

        parsed_tweets = []
        for dirty_tweet in tweet_data:
            parsed_tweets.append(Tokenizer(dirty_tweet)

class TwitterStreamer(tweepy.StreamListener):

    def __init__(self, tweet_file, twitter_api=None):
        super(TwitterStreamer, self).__init__()
        self.file_location = tweet_file
        self.num_of_tweets = 0

    def on_data(self, data):
        print("Entered on data")
        if self.num_of_tweets < 15000:
            try:
                with open(self.file_location, 'a') as tweets:
                    tweets.write(data)
                    self.num_of_tweets += 1
                    return True
            except BaseException as bex:
                print("Error on_data: " + str(bex))

    def on_error(self, status):
        if status == 420:
            # returning False in on_data disconnects the stream
            return False
        return True
