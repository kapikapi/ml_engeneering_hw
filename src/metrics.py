import numpy as np
import pandas as pd


def _p_k(y_true, y_pred, k):
    return len(np.intersect1d(y_pred[:k], y_true[:k])) / k


def map_k(y_true, y_pred, k):
    res = 0
    for i in range(k):
        p_i = _p_k(y_true, y_pred, i + 1)
        r_true = 1 if y_pred[i] in y_true[:k] else 0
        res = res + r_true * p_i
    return res / k


def hit_rate_k(y_true, y_pred, k):
    return 1 if len(np.intersect1d(y_pred[:k], y_true[:k])) > 0 else 0


def ndcg_k(y_true, y_pred, k):
    dcg_k = 0
    for i in range(k):
        r_true = 1 if y_pred[i] in y_true[:k] else 0
        dcg_k = dcg_k + r_true * np.log2(i + 1)
    idcg_k = np.sum([1 / np.log2(i + 2) for i in range(k)])
    return dcg_k / idcg_k


def avg_metric(y_true, y_pred, metric, k=None):
    merged = pd.merge(y_true, y_pred, on='user_id', how='right', suffixes=('_true', '_pred'))
    merged['recs_true'] = merged['recs_true'].apply(lambda x: x if isinstance(x, list) else [])
    l = [metric(row['recs_true'], row['recs_pred'], k) for idx, row in merged.iterrows()]
    return np.mean(l)


def print_metrics(true_res, predicted_res, k_values=(3, 5, 10)):
    str_metrics = ''
    dict_metrics = {}
    for k in k_values:
        hit_rate_avg = round(avg_metric(true_res, predicted_res, hit_rate_k, k), 3)
        map_avg = round(avg_metric(true_res, predicted_res, map_k, k), 3)
        ndcg_avg = round(avg_metric(true_res, predicted_res, ndcg_k, k), 3)
        metric_line = f'HitRate@{k}: {hit_rate_avg}; MAP@{k}: {map_avg}; NDCG@{k}: {ndcg_avg};'
        str_metrics += metric_line + '\n'
        dict_metrics[f"HitRate@{k}"] = hit_rate_avg
        dict_metrics[f"MAP@{k}"] = map_avg
        dict_metrics[f"NDCG@{k}"] = ndcg_avg
    return str_metrics, dict_metrics
