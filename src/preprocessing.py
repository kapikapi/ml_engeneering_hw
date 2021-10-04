import sys
import os
import pandas as pd

input_train = sys.argv[1]
input_test = sys.argv[2]
input_orders = sys.argv[3]
output_train = os.path.join("data", "prepared", "train_table.csv")
output_test = os.path.join("data", "prepared", "test_table.csv")


def preprocess_orders(input_paths, output_paths):
    os.makedirs(os.path.dirname(output_train))
    train = pd.read_csv(input_paths[0])
    test = pd.read_csv(input_paths[1])
    orders = pd.read_csv(input_paths[2])
    train_data = pd.merge(train, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    test_data = pd.merge(test, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    train_data.to_csv(output_paths[0])
    test_data.to_csv(output_paths[1])


preprocess_orders((input_train, input_test, input_orders), (output_train, output_test))
