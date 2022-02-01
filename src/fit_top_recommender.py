import os
import pickle
from collections import Counter

import pandas as pd

top_columns = {'item': 'product_id'}


def fit_top_recommender(train_data_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    train_data = pd.read_csv(train_data_path)
    top_recommender = Counter(train_data[top_columns['item']]).most_common()
    with open(output_folder + "/model.pickle", "wb") as output_file:
        pickle.dump(top_recommender, output_file)






