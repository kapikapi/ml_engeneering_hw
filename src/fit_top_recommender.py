import os
import sys
import pickle
from collections import Counter

import pandas as pd

top_columns = {'item': 'product_id'}


def fit_top_recommender(output_folder, train_data_path, cols):
    train_data = pd.read_csv(train_data_path)
    top_recommender = Counter(train_data[cols['item']]).most_common()
    with open(output_folder + "/model.pickle", "wb") as output_file:
        pickle.dump(top_recommender, output_file)


train_path = sys.argv[1]
os.makedirs(sys.argv[2], exist_ok=True)
output_folder_path = sys.argv[2]

fit_top_recommender(output_folder_path, train_path, top_columns)





