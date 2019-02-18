import warnings

from DataPresentation import DataPresentation
from train_model import TrainModels
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dp = DataPresentation()
    data = dp.read_csv()
    dp.data_to_df(data)
   # dp.print_common()
   # tm = TrainModels(dp.data_frame)
   # best_model = tm.train_models()
   # print(best_model)

    # Find the country with the most tweets
    train_data, test_data = train_test_split(data, test_size=0.2)
    most_common_countries = dp.find_most_common_country(train_data)
    for country in most_common_countries:
        print(country)

    tweets_location = '.\\tweets_from_stream.json'
    dp.collect_tweets(file_location=tweets_location)
    dp.process_tweets(tweets_location)
