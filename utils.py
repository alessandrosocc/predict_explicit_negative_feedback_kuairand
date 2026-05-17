SEED = 2026

import os
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from functools import wraps
from time import perf_counter
from typing import Callable

def split_group_by_user(
    df: pd.DataFrame,
    seed: int = SEED,
):

    rng = np.random.default_rng(seed)

    train_parts = []
    test_parts = []
    for _, user_df in df.groupby('user_id', sort=False):
        user_df = user_df.sample(frac=1.0, random_state=rng)

        pos_df = user_df[user_df["label"] == 1]
        neg_df = user_df[user_df["label"] == 0]
        
        assert len(pos_df)>0, f"User {user_df['user_id'].iloc[0]} has no positive interactions"
        assert len(neg_df)>0, f"User {user_df['user_id'].iloc[0]} has no negative interactions"

        pos_test = pos_df.sample(n=1, random_state=rng)
        neg_test = neg_df.sample(n=1, random_state=rng)
        test_part = pd.concat([pos_test, neg_test])
        train_part = user_df.drop(test_part.index)

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df.drop(columns=["timestamp"], inplace=True)
    test_df.drop(columns=["timestamp"], inplace=True)

    return train_df, test_df



# def make_chunks(df: pd.DataFrame, chunk_size: int = 100_000):
#     data, label = list(), list()
#     for i in range(0, len(df), chunk_size):
#         chunk = df.iloc[i:i + chunk_size]
#         data.append(chunk.drop(columns=["label"]).to_numpy())
#         label.append(chunk[["label"]].to_numpy())

#     return data, label


def get_dataset(args):
    try:
        return pl.read_csv(
            os.path.join("processed_data", f"processed_{args.policy}_kcore_{args.k_core}.csv"),
            separator="\t",
        ).to_pandas()
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Please run the preprocessing script first.")
    


def show_results(args, model_name, true_labels, preds, split_name):
    print(f"{split_name} results:")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, preds, labels=[0, 1], zero_division=0
    )
    print(f"{'Metric':<10} {'0 (unliked)':>10} {'1 (liked)':>12}")
    print(f"{'Recall':<10} {recall[0]:>10.4f} {recall[1]:>12.4f}")
    print(f"{'Precision':<10} {precision[0]:>10.4f} {precision[1]:>12.4f}")
    print(f"{'F1 Score':<10} {f1[0]:>10.4f} {f1[1]:>12.4f}")
    print("----------------------------------")
    print(f"{'Support':<10} {support[0]:>10d} {support[1]:>12d}")
    print()

    # save to csv the results in a folder called results with filename {model}_{policy}_kcore_{k_core}_results.csv
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame({
        "metric": ["precision", "recall", "f1_score", "support"],
        "0 (unliked)": [precision[0], recall[0], f1[0], support[0]],
        "1 (liked)": [precision[1], recall[1], f1[1], support[1]],
    })
    if model_name in ['finetune_tabpfn3.0']:
        model_name = f'finetune_tabpfn3.0_{args.epochs}_epochs_'
    results_df.to_csv(os.path.join("results", f"{model_name}_{args.policy}_kcore_{args.k_core}_results.csv"), index=False, sep='\t')



# def _concat_chunks(chunks, labels):
#     features = np.concatenate(chunks, axis=0)
#     targets = np.concatenate(labels, axis=0).ravel()
#     return features, targets


def vectorize_data(df):
    features = df.drop(columns=["label"]).to_numpy()
    targets = df[["label"]].to_numpy().ravel()
    return features, targets


def timer(func: Callable) -> Callable:
    def _format_elapsed(seconds: float) -> str:
        seconds_int = int(round(seconds))
        hours, remainder = divmod(seconds_int, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        return results, _format_elapsed(run_time)

    return wrapper