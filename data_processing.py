import polars as pl
import argparse
import os

def _make_user_k_core(df, args):
    # make user k-core
    valid_user_ids = (
        df["user_id"]
        .value_counts()
        .filter(pl.col("count") >= args.k_core)["user_id"]
        .implode()
    )
    return df.filter(pl.col("user_id").is_in(valid_user_ids))

def _make_item_k_core(df, args):
    # make item k-core
    valid_video_ids = (
        df["video_id"]
        .value_counts()
        .filter(pl.col("count") >= args.k_core)["video_id"]
        .implode()
    )
    return df.filter(pl.col("video_id").is_in(valid_video_ids))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--policy", type=str, default="standard", help="The policy to use for data processing. Options: standard, random")
    args.add_argument("--k_core", type=int, default=10, help="The minimum number of interactions for a user or item to be included")
    args = args.parse_args()

    if args.policy == "standard":
        df_1 = pl.read_csv(os.path.join('KuaiRand-27K', 'log_standard_4_08_to_4_21_27k_part1.csv'))
        df_2 = pl.read_csv(os.path.join('KuaiRand-27K', 'log_standard_4_08_to_4_21_27k_part2.csv'))
        df = pl.concat([df_1, df_2], how='vertical')
    elif args.policy == "random":
        df = pl.read_csv(os.path.join('KuaiRand-27K', 'log_random_4_22_to_5_08_27k.csv'))
    else:
        raise ValueError("Invalid policy. Options: normal, random")

    
    # remove is_like or is_hate and create a new column "label" where is_like is 1 and is_hate is 0 then label is 1, if is_like is 0 and is_hate is 1 then label is 0, otherwise drop the row
    df = df.filter((pl.col("is_like") == 1) | (pl.col("is_hate") == 1))
    df = df.with_columns(
        pl.when(pl.col("is_like") == 1)
        .then(1)
        .when(pl.col("is_hate") == 1)
        .then(0)
        .otherwise(pl.lit(None))
        .cast(pl.Int64)
        .alias("label")
    )
    df = df.drop(["is_like", "is_hate"])

    # select all users that have at least one positive feedback and one negative feedback, then keep only interactions of those users. This is to ensure that we have both positive and negative feedback for each user in the dataset, which is important for training a model to predict negative feedback.
    valid_user_ids = (
        df.group_by("user_id")
        .agg(pl.col("label").n_unique().alias("label_types"))
        .filter(pl.col("label_types") == 2)
        .select("user_id")
    )
    df = df.join(valid_user_ids, on="user_id", how="inner")
    

    print(f"size before k-core: {df.shape}")
    print(f'positive feedbacks before k-core: {df.filter(pl.col("label") == 1).shape[0]}')
    print(f'negative feedbacks before k-core: {df.filter(pl.col("label") == 0).shape[0]}')

    # filter to only where label is 1
    positive_df = df.filter(pl.col("label") == 1).clone()
    positive_df = _make_item_k_core(positive_df, args)
    positive_df = _make_user_k_core(positive_df, args)
    # concat negative samples where label==0 with positive_df
    negative_df = df.filter(pl.col("label") == 0)
    df = pl.concat([positive_df, negative_df], how='vertical')
    
    print()
    print(f"size after k-core: {df.shape}")
    print(f'positive feedbacks after k-core: {df.filter(pl.col("label") == 1).shape[0]}')
    print(f'negative feedbacks after k-core: {df.filter(pl.col("label") == 0).shape[0]}')

    # remove useless columns
    df = df.drop(["profile_stay_time", "is_rand", "tab"])


    # create a readable timestamp string from time_ms
    df = df.with_columns(
        pl.from_epoch("time_ms", time_unit="ms")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("timestamp")
    )


    # ensure all features are i64 or f64
    for col in df.columns:
        if col not in ['timestamp'] and df[col].dtype not in [pl.Int64, pl.Float64]:
            raise ValueError(f"Column {col} has invalid dtype {df[col].dtype}. All features must be i64 or f64.")
    
    
    os.makedirs("processed_data", exist_ok=True)
    
    print(f"size: {df.shape}")
    print(f"num_users: {df['user_id'].n_unique()}")
    print(f"num_videos: {df['video_id'].n_unique()}")

    # save polars processed data
    df.write_csv(os.path.join("processed_data", f"processed_{args.policy}_kcore_{args.k_core}.csv"), separator="\t")