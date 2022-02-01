from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from preprocessing import preprocess_orders
from fit_svd import fit_svd_recommender
from fit_top_recommender import fit_top_recommender
from predict_svd import predict_svd
from predict_top import predict_top
from metrics import print_metrics

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
with DAG(
    'rec_sys',
    default_args=default_args,
    description='Recommendation DAG',
    schedule_interval="@once",
    start_date=datetime(2022, 1, 25),
    catchup=False,
    tags=['example_tag'],
) as dag:

    dummy_task = DummyOperator(task_id="dummy_task", dag=dag)
    preprocessing_task = PythonOperator(task_id="preprocessing_task",
                                        python_callable=preprocess_orders,
                                        op_args=["data/train.csv", "data/test.csv", "data/orders.csv"], dag=dag)

    fit_svd_task = PythonOperator(task_id="fit_svd_task",
                                  python_callable=fit_svd_recommender,
                                  op_args=["data/prepared/train_table.csv", "static/svd_model"])

    fit_top_task = PythonOperator(task_id="fit_svd_task",
                                  python_callable=fit_top_recommender,
                                  op_args=["data/prepared/train_table.csv", "static/top_rec_model"])

    predict_svd_task = PythonOperator(task_id="predict_svd_task",
                                      python_callable=predict_svd,
                                      op_args=["data/prepared/test_table.csv",
                                               ("static/svd_model/users_repres.npy",
                                                "static/svd_model/products_repres.npy",
                                                "static/svd_model/user_pos.json",
                                                "static/svd_model/pos_product.json"),
                                               "static/recs/svd"])
    predict_top_task = PythonOperator(task_id="predict_top_task",
                                      python_callable=predict_top,
                                      op_args=["static/recs/top",
                                               "data/prepared/test_table.csv",
                                               "static/top_rec_model/model.pickle"])

    metrics_task = PythonOperator(task_id="metrics_task",
                                  python_callable=print_metrics,
                                  op_args=["data/prepared/test_table.csv",
                                           "static/top_rec_model/model.pickle",
                                           "static/recs/top"])


    dummy_task >> preprocessing_task >> [fit_svd_task, fit_top_task]
    fit_svd_task >> predict_svd_task
    fit_top_task >> predict_top_task
    [predict_svd_task, predict_top_task] >> metrics_task

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    # t1 = BashOperator(
    #     task_id='print_date',
    #     bash_command='date',
    # )
    #
    # t2 = BashOperator(
    #     task_id='sleep',
    #     depends_on_past=False,
    #     bash_command='sleep 5',
    #     retries=3,
    # )
    #
    # templated_command = dedent(
    #     """
    # {% for i in range(5) %}
    #     echo "{{ ds }}"
    #     echo "{{ macros.ds_add(ds, 7)}}"
    #     echo "{{ params.my_param }}"
    # {% endfor %}
    # """
    # )
    #
    # t3 = BashOperator(
    #     task_id='templated',
    #     depends_on_past=False,
    #     bash_command=templated_command,
    #     params={'my_param': 'Parameter I passed in'},
    # )
    #
    # t1 >> [t2, t3]