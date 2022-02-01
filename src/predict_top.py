import json

import pandas as pd
import yaml
import pickle
import os
from metrics import print_metrics

params = yaml.safe_load(open("params.yaml"))["predict_top"]
n_recs = params["n_recs"]


def predict_top(output_folder, test_data_path, model_path):
    os.makedirs(output_folder, exist_ok=True)
    test_data = pd.read_csv(test_data_path)
    test_users = list(set(test_data.user_id))
    true_recs = test_data.groupby('user_id')['product_id'].apply(list).reset_index(name='recs')

    with open(model_path, "rb") as model_file:
        top_model = pickle.load(model_file)

    most_popular_items = [x[0] for x in top_model[:n_recs]]
    most_popular_items_users = [most_popular_items] * len(test_users)

    recs = pd.DataFrame({'user_id': test_users, 'recs': most_popular_items_users},
                        columns=['user_id', 'recs']).sort_values('user_id').reset_index(drop=True)
    recs.to_csv(output_folder + "/top_recs.csv")

    str_metrics, dict_metrics = print_metrics(true_recs, recs)
    with open(output_folder + "/top_metrics.txt", "w") as text_file:
        text_file.write(str_metrics)

    with open(output_folder + "/top_metrics.json", "w") as metrics_file:
        json.dump(dict_metrics, metrics_file)






