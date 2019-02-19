import warnings

from DataPresentation import DataPresentation
from train_model import TrainModels
from sklearn.model_selection import train_test_split
from keras.models import load_model, save_model

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dp = DataPresentation()
    data = dp.read_csv(print_graph=False)
    #dp.data_to_df(data)
    #dp.print_common()
    #tm = TrainModels(dp.data_frame)
    #best_model = tm.train_models()
    #print(best_model)
    #best_model.save('.\\best_model.h5')
    # Find the country with the most tweets
    #train_data, test_data = train_test_split(data, test_size=0.2)
    #most_common_countries = dp.find_most_common_country(train_data, 5)
    #for country in most_common_countries:
    #    print(country)

    tweets_location = '.\\tweets_from_stream.json'
    #dp.collect_tweets(file_location=tweets_location)
    parsed_tweets, tweet_distribution = dp.process_tweets(tweets_location, filter_words=True)
    #most_common = dp.get_most_common_words(tweet_distribution)

    #for i in range(30):
    #    print(f'{i+1}. {most_common[i][0]}, {most_common[i][1]} Times ')

    model = load_model('.\\best_model.h5')
    results_file = '.\\prediction_results.csv'
    with open(results_file, 'w') as res:
        header = "tweet,prediction,confidence\n"
        res.write(header)

    predictions = dp.predict_gender(model, parsed_tweets)

    with open(results_file, 'a') as res:
        for idx in range(len(predictions)):
            res.write(f'{parsed_tweets[idx]}, {predictions[idx]}, {idx}\n')

    results = dp.aggregate_predictions(predictions)
    for res, num in results.items():
        print(f'Number of {res}s: {num}')
