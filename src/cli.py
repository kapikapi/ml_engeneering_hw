import mlflow

from src.fit_svd import fit_svd_recommender
# from src.fit_top_recommender import fit_top_recommender
from src.predict_svd import predict_svd
from src.preprocessing import preprocess_orders
from mlflow.tracking import MlflowClient

# mlflow.set_tracking_uri("gs://try-mlflow-bucket")
mlflow.set_tracking_uri("")
experiment_id = mlflow.create_experiment("experimant_gcs_1")

# client = MlflowClient()
# experiments = client.list_experiments()
# experiment_id = experiments[0].experiment_id

train_data_path = "../data/train.csv"
test_data_path = "../data/test.csv"
orders_data_path = "../data/orders.csv"

preprocessed_train_path = "../data/prepared/train_table.csv"
preprocessed_test_path = "../data/prepared/test_table.csv"


def train_with_params(n_comps, batches, num_recs, metric="HitRate_10"):
    best_result = 0
    best_n_comps = -1
    best_batches = -1
    best_results = None
    columns = {'user': 'user_id', 'item': 'product_id', 'group_id': 'order_id'}
    outputs = ["../static/svd_model/users_repres.npy", "../static/svd_model/products_repres.npy",
               "../static/svd_model/user_pos.json", "../static/svd_model/pos_product.json", ""]
    for n in n_comps:
        for b in batches:
            fit_svd_recommender("../static/svd_model", preprocessed_train_path, columns, n_comp=n, random_state=0)

            svd_metrics = predict_svd("../static/recs/svd", "../data/prepared/test_table.csv", outputs, n=num_recs,
                                      batch_num=b)
            m = svd_metrics[metric]
            if m > best_result:
                best_results = svd_metrics
                best_n_comps = n
                best_batches = b
                print(f"Current best params are {b} as batch size and {n} as number of components. {metric}={m}")
    mlflow.log_params({"number of recs": num_recs, "batch size": best_batches, "num components": best_n_comps})
    mlflow.log_metrics(best_results)
    mlflow.log_artifacts("../static/svd_model")


with mlflow.start_run(experiment_id=experiment_id):
    preprocess_orders((train_data_path, test_data_path, orders_data_path), (preprocessed_train_path, preprocessed_test_path))
    # n_comps = [250, 500, 750]
    # batch_nums = [100, 150]
    n_comps = [200, 750]
    batch_nums = [100]
    num_recs = 10
    train_with_params(n_comps, batch_nums, num_recs, "HitRate_10")

