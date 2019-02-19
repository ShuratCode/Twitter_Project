import operator
import re
import string
from collections import Counter
import pprint
import tweepy
import json

from keras_preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy
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

    def read_csv(self, print_graph=True):
        """
        Read the csv file and present it
        :return: representation of the
        """
        data = pd.read_csv(self.__DB_path, encoding='latin-1')
        if print_graph:
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
        return self.data_frame

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
    def find_most_common_country(data, top):

        countries = data['tweet_location'].value_counts()
        top_countries = []
        for loc, num in countries.items():
            top_countries.append((loc, num))
        if top < len(top_countries):
            return top_countries[0:top]
        else:
            return top_countries

    @staticmethod
    def collect_tweets(file_location):

        api_key = "GXRCrOUdw2q35n8DduLSWYCGL"
        api_secret_key = "Ynl09Fgy07LqPVyHPkx0NlodNoqyEuP3JvCSdIR1vKseszxdM3"

        access_token = "734660944920465408-ORnK69HUaAKirszN1pW12XfzBGyghrB"
        access_token_secret = "WrgnSATTGPVP7yJVQvCmC0vqZtGUHAtzTgqRcmQ3Uhvyo"

        authenticator = tweepy.OAuthHandler(api_key, api_secret_key)
        authenticator.set_access_token(access_token, access_token_secret)
        twitter_api = tweepy.API(authenticator)

        stream_listener = TwitterStreamer(file_location)
        my_twitter_stream = tweepy.Stream(auth=twitter_api.auth, listener=stream_listener)
        my_twitter_stream.filter(locations=[-0.510375, 51.28676, 0.334015, 51.691874], is_async=True)

    def process_tweets(self, file_location, filter_words=True):
        """
        Processes the tweets in the same manner as in Question 1
        :param filter_words:
        :param file_location: Location of the tweets json
        """
        tweet_data = []
        tweets_json = open(file_location, 'r')

        for line in tweets_json:
            if line == '\n':
                continue
            try:
                curr_tweet = json.loads(line)
                tweet_data.append(curr_tweet['text'])
            except Exception as ex:
                #print(ex)
                continue

        parsed_tweets = []
        for dirty_tweet in tweet_data:
            tweet_token = self.preprocess(dirty_tweet, lowercase=True)
            parsed_tweets.append(tweet_token)

        clean_tweets = []
        term_freq = {}
        punctuations = {'\'', '\"', '\\', '/', '`', '.', '!', ';', '&', '(', ')', ',', '?', '-', ':', '', '@'}
        stop_words = set(stopwords.words("english"))
        for parsed_tweet in parsed_tweets:
            txt = ''
            for token in parsed_tweet:
                txt = txt + token + " "  # Todo: concatenate the dirty token or parsed token?
                if filter_words and (token in punctuations or token in stop_words):
                    continue
                token = self.ps.stem(token)
                if token not in term_freq:
                    term_freq[token] = 1
                else:
                    term_freq[token] += 1

            clean_tweets.append(txt)

        return clean_tweets, term_freq

    @staticmethod
    def get_most_common_words(tweet_distribution):

        sorted_tweets = []
        for key, value in tweet_distribution.items():
            sorted_tweets.append((key, value))

        sorted_tweets.sort(key=operator.itemgetter(1), reverse=True)
        return sorted_tweets

    def predict_gender(self, model, parsed_tweets):
        """
        Receives a list of parsed tweets and predicts for each one the gender of the user
        :param model: A classifier
        :param parsed_tweets: A list of parsed tweets
        :return: The prediction for each tweet.
        """
        tweet_array = numpy.asarray(parsed_tweets)
        tokenizer = Tokenizer(num_words=2000)
        tweet_matrix = tokenizer.texts_to_matrix(tweet_array, mode='binary')
        predictions = model.predict_classes(tweet_matrix)

        return predictions

    @staticmethod
    def aggregate_predictions(predictions):

        results = {
            "Male": 0,
            "Female": 0,
            "Brand": 0
        }

        for pred in predictions:
            results[pred] += 1

        return results


class TwitterStreamer(tweepy.StreamListener):
    """
    Listener class inheriting from tweepy's StreamListener class.
    """

    def __init__(self, tweet_file):
        super(TwitterStreamer, self).__init__()
        self.file_location = tweet_file
        self.num_of_tweets = 0

    def on_connect(self):
        print("connected")
        return True

    def on_data(self, data):
        if self.num_of_tweets < 15000:
            try:
                with open(self.file_location, 'a') as tweets:
                    tweets.write(data)
                    self.num_of_tweets += 1
                    if self.num_of_tweets % 100 == 0:
                        print(f'Collected {self.num_of_tweets} tweets')

                    return True
            except BaseException as bex:
                print("Error on_data: " + str(bex))

    def on_error(self, status):
        if status == 420:
            # returning False in on_data disconnects the stream
            return False
        return True
