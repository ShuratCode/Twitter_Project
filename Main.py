import warnings

from DataPresentation import DataPresentation
from train_model import TrainModels

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dp = DataPresentation()
    data = dp.read_csv()
    dp.data_to_df(data)
    dp.print_common()
    tm = TrainModels(dp.data_frame)
    best_model = tm.train_models()
    print(best_model)
