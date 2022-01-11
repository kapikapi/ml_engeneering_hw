import os
import pandas as pd


def preprocess_orders(input_paths, output_paths):
    os.makedirs(os.path.dirname("data/prepared"), exist_ok=True)
    train = pd.read_csv(input_paths[0])
    test = pd.read_csv(input_paths[1])
    orders = pd.read_csv(input_paths[2])
    train_data = pd.merge(train, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    test_data = pd.merge(test, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    train_data.to_csv(output_paths[0])
    test_data.to_csv(output_paths[1])
