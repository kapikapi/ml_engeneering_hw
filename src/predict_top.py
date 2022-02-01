import pandas as pd
import yaml
import pickle
import os


def predict_top(output_folder, test_data_path, model_path):
    params = yaml.safe_load(open("dags/params.yaml"))["predict_top"]
    n_recs = params["n_recs"]

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







