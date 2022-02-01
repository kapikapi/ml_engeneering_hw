import sys
import os
import pandas as pd

output_train = os.path.join("data", "prepared", "train_table.csv")
output_test = os.path.join("data", "prepared", "test_table.csv")


def preprocess_orders(input_train, input_test, input_orders):
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    train = pd.read_csv(input_train)
    test = pd.read_csv(input_test)
    orders = pd.read_csv(input_orders)
    train_data = pd.merge(train, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    test_data = pd.merge(test, orders, on='order_id', how='inner')[['order_id', 'product_id', 'user_id']]
    train_data.to_csv(output_train)
    test_data.to_csv(output_test)
