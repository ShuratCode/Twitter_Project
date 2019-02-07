import csv
import re
import string

from nltk.corpus import stopwords

DB_path = './gender-classifier-DFE-791531.csv'
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def read_csv():
    db_csv = open(DB_path, 'r', encoding='utf8', errors='ignore')
    reader = csv.reader(db_csv, delimiter=',')
    headers = next(reader)
    text_idx = headers.index('text')
    gender_idx = headers.index('gender')
    terms_male, terms_female, terms_brand = build_terms(text_idx, gender_idx, reader)
    # TODO: train models on this data.


def build_terms(text_idx, gender_idx, reader):
    terms_male = []
    terms_female = []
    terms_brand = []
    punctuation = list(string.punctuation)
    stop_words = stopwords.words('english') + punctuation + ['rt', 'via']

    for line in reader:
        tokens = preprocess(line[text_idx])
        for token in tokens:
            if token not in stop_words:
                if line[gender_idx] == 'male':
                    terms_male.append(token)
                if line[gender_idx] == 'female':
                    terms_female.append(token)
                if line[gender_idx] == 'brand':
                    terms_brand.append(token)

    return terms_male, terms_female, terms_brand


def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
