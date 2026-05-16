import argparse
import os
from utils import SEED
from utils import get_dataset, split_group_by_user, vectorize_data, show_results, timer
import setproctitle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def get_model(args, model_name):
    # tabpfn2.0 supports only up to 10k samples.
    # if model_name == "tabpfn2.0":
    #     from tabpfn import TabPFNClassifier
    #     from tabpfn.constants import ModelVersion
    #     return TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=SEED, balance_probabilities = True)
    if model_name == "tabpfn2.5":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, random_state=SEED, balance_probabilities = True)
    elif model_name == "tabpfn2.6":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2_6, random_state=SEED, balance_probabilities = True)
    elif model_name == "tabpfn3.0":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V3, random_state=SEED, balance_probabilities = True)
    elif model_name == 'finetune_tabpfn3.0':
        from tabpfn.finetuning import FinetunedTabPFNClassifier
        return FinetunedTabPFNClassifier(
            device="cuda",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=SEED, device='cuda')
    elif model_name == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_state=SEED)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier()
    elif model_name == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=SEED)
    elif model_name == 'random':
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy='uniform', random_state=SEED)
    elif model_name == 'tabicl':
        from tabicl import TabICLClassifier
        return TabICLClassifier(random_state=SEED)
    else:
        raise ValueError("Invalid model name. Options: tabpfn, xgboost, lightgbm, catboost, random forest")



@timer
def train(model, data):
    train_data, train_labels = data

    model = model.fit(train_data, train_labels)

    return model

@timer
def evaluate(model, data):
    test_data, test_labels = data

    test_preds = model.predict(test_data)

    return test_preds, test_labels


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="tabpfn", help="The model to train. Options: tabpfn, xgboost, lightgbm")
    args.add_argument("--policy", type=str, default="standard", help="The policy to use. Options: standard, random")
    args.add_argument("--k_core", type=int, default=5, help="The k-core to use.")
    # in case of finetuning tabpfn3.0, we want to specify the number of epochs and learning rate
    args.add_argument("--epochs", type=int, default=5, help="The number of epochs to finetune tabpfn3.0")
    args.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate to finetune tabpfn3.0")
    # explainability
    args.add_argument("--explain", type=bool, default=False, help="Whether to compute SHAP values for tabpfn models (only for tabpfn2.5, tabpfn2.6 and tabpfn3.0)")
    args = args.parse_args()

    setproctitle.setproctitle(f"Predicting explicit negative feedback in short-video recsys with {args.model}")

    dataset = get_dataset(args)
    
    track_time = defaultdict(dict)

    for model_name in args.model.split(","):
        print(f"Training model: {model_name}")
        model = get_model(args, model_name)

        # Split the dataset into train, validation, and test sets, grouped by user id
        train_df, test_df = split_group_by_user(dataset, ratios=(0.6, 0.2, 0.2))


        print("Train set size:", len(train_df))
        print("Test set size:", len(test_df))


        # Keep feature names before vectorization (numpy arrays do not have .columns).
        feature_names = [c for c in test_df.columns if c != "label"]

        # Create chunks first, then merge into single arrays for model APIs.
        train_data, train_labels = vectorize_data(train_df)
        test_data, test_labels = vectorize_data(test_df)


        # Training
        data = (train_data, train_labels)

        model, time = train(model, data)
        print("Training completed.")
        track_time[model_name]["train"] = str(time)

        # Evaluation on validation and test sets
        data = (test_data, test_labels)
        res, time = evaluate(model, data)
        preds, labels = res

        track_time[model_name]["test"] = str(time)

        show_results(args, model_name, labels, preds, split_name="Test")
        print("Evaluation completed.")

        if model_name in ['tabpfn3.0', 'tabpfn2.6', 'tabpfn2.5'] and args.explain:
            from tabpfn_extensions.interpretability.shap import get_shap_values, plot_shap
            import shap

            # select randomly 100 samples from the test set to compute SHAP values
            # too slow, not usable in practice.
            test_data = test_data[:100]
            test_labels = test_labels[:100]
            shap_values = get_shap_values(
                estimator=model,
                test_x=test_data,
                attribute_names=feature_names,
            )
            print("SHAP values computed.")

            # Full interpretability view (bar + summary + interaction plot), as in docs.
            plot_shap(shap_values)

            # Save the aggregate bar plot to file (same style as docs bar chart).
            os.makedirs("results/plots", exist_ok=True)
            bar_values = shap_values[:, :, 0] if len(shap_values.shape) == 3 else shap_values
            shap.plots.bar(shap_values=bar_values, show=False)
            plt.title("Aggregate feature importances across the test examples")
            plt.tight_layout()
            plt.savefig(
                os.path.join("results", "plots", f"feature_importance_shap_{model_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
    
    # create a dataframe from track_time that contain a column for the model, the split (train or test) and the time taken, then save it to a csv file in the results folder with filename {model}_{policy}_kcore_{k_core}_time.csv
    time_df = pd.DataFrame([
        {"model": model_name, "split": split, "time": time} for model_name, splits in track_time.items() for split, time in splits.items()
    ])
    time_df.to_csv(os.path.join("results", f"time_{args.policy}_kcore_{args.k_core}.csv"), index=False, sep='\t')

        
