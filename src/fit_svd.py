import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

import numpy as np
import json
import os

params = yaml.safe_load(open("dags/params.yaml"))["fit_svd"]
n_components = params["n_components"]
random_seed = params["random_seed"]

def _get_counts_df(df, column_names):
    return (df
            .groupby([column_names['user'], column_names['item']])[[column_names['group_id']]]
            .count()
            .reset_index()
            .rename(columns={column_names['group_id']: "count"}))


def _get_interactions_matrix(train_data, column_names, user_pos, pos_user, product_pos, pos_product):
    train_data_count = _get_counts_df(train_data, column_names)
    rows = [user_pos[user] for user in train_data_count[column_names['user']].values]
    cols = [product_pos[product] for product in train_data_count[column_names['item']].values]
    return csr_matrix((train_data_count['count'].values, (rows, cols)),
                      shape=(len(pos_user.keys()), len(pos_product.keys())))


def fit_svd_recommender(train_data_path, output_folder):
    svd_columns = {'user': 'user_id', 'item': 'product_id', 'group_id': 'order_id'}

    os.makedirs(output_folder, exist_ok=True)
    train_data = pd.read_csv(train_data_path)
    train_users = set(train_data.user_id)
    train_products = set(train_data.product_id)

    user_pos = {user: idx for idx, user in enumerate(train_users)}
    pos_user = {value: key for key, value in user_pos.items()}
    product_pos = {product: idx for idx, product in enumerate(train_products)}
    pos_product = {value: key for key, value in product_pos.items()}

    interactions_matrix = _get_interactions_matrix(train_data, svd_columns, user_pos=user_pos, pos_user=pos_user,
                                                   product_pos=product_pos, pos_product=pos_product)
    svd_recommender = TruncatedSVD(random_state=random_seed, n_components=n_components)
    users_repres = svd_recommender.fit_transform(interactions_matrix)
    products_repres = svd_recommender.components_
    print(f'User representaions have size: {users_repres.shape}')
    print(f'Item representaions have size: {products_repres.shape}')

    np.save(output_folder + "/users_repres.npy", users_repres)
    np.save(output_folder + "/products_repres.npy", products_repres)

    with open(output_folder + "/user_pos.json", "w") as write_file:
        json.dump(user_pos, write_file)
    with open(output_folder + "/pos_product.json", "w") as write_file:
        json.dump(pos_product, write_file)






