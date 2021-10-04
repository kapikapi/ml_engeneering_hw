import pandas as pd
import yaml
import numpy as np
import json
import os
import sys
from metrics import print_metrics

params = yaml.safe_load(open("params.yaml"))["predict_svd"]
n_recs = params["n_recs"]
batch_number = params["batch_number"]


def _predict_batch(test_users_batch, n, users_repres, products_repres, user_pos, pos_user, pos_product):
    recs = []
    batch_users_pos = [user_pos[user] for user in test_users_batch]
    batch_user_ratings = np.dot(users_repres[batch_users_pos, :], products_repres)
    sorted_recs = batch_user_ratings.argsort()[:, ::-1][:, :n]
    for i, user in enumerate(batch_users_pos):
        recs_dict = {'user_id': pos_user[user],
                     'recs': [pos_product[rec] for rec in sorted_recs[i, :]]}
        recs.append(recs_dict)

    return recs


def predict_svd(output_folder, test_data_path, model_paths, n, batch_num):
    users_repres = np.load(model_paths[0])
    products_repres = np.load(model_paths[1])

    with open(model_paths[2]) as user_pos_json:
        user_pos = {int(k): v for k, v in json.load(user_pos_json).items()}
    pos_user = {value: key for key, value in user_pos.items()}

    with open(model_paths[3]) as pos_product_json:
        pos_product = {int(k): v for k, v in json.load(pos_product_json).items()}

    test_data = pd.read_csv(test_data_path)
    test_users = list(set(test_data.user_id))
    true_recs = test_data.groupby('user_id')['product_id'].apply(list).reset_index(name='recs')

    batches_users = np.array_split(test_users, batch_num)
    result_list = []
    for batch in batches_users:
        predicted_batch = _predict_batch(batch, n, users_repres=users_repres, products_repres=products_repres,
                                         user_pos=user_pos, pos_user=pos_user, pos_product=pos_product)
        result_list += predicted_batch

    recs = pd.DataFrame.from_records(result_list).sort_values('user_id').reset_index(drop=True)
    recs.to_csv(output_folder + "/svd_recs.csv")

    metrics = print_metrics(true_recs, recs)
    with open(output_folder + "/svd_metrics.txt", "w") as text_file:
        text_file.write(metrics)


test_data_pth = sys.argv[1]
model_pths = sys.argv[2:-1]
output_pth = sys.argv[-1]
os.makedirs(output_pth, exist_ok=True)

predict_svd(output_pth, test_data_pth, model_pths, n_recs, batch_number)





