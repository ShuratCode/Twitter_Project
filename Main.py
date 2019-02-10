fr
import warnings
from collections import Counter

from DataPresentation import DataPresentation

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dp = DataPresentation()
    data = dp.read_csv()
    dp.data_to_df(data)
    dp.print_common()

