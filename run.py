import argparse
import os
from utils import SEED
from utils import get_dataset, split_group_by_user, vectorize_data, show_results, timer
import setproctitle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re


def _format_feature_row(row, feature_names):
    def _format_feature_value(value):
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value) == 0.0:
                return "false"
            if float(value) == 1.0:
                return "true"
        return str(value)

    if feature_names and len(feature_names) == len(row):
        return "\n".join(f"- {name}: {_format_feature_value(value)}" for name, value in zip(feature_names, row))
    return "\n".join(f"- f{i}: {_format_feature_value(value)}" for i, value in enumerate(row))


def _label_to_int(text):
    if text.startswith("0"):
        return 0
    return 1


def _build_binary_feedback_prompt(row, feature_names):
    row_repr = _format_feature_row(row, feature_names)
    return (
        "Features:\n"
        f"{row_repr}\n"
        "\n"
        "Task: binary classification (user liked item).\n"
        "Rules: return only 1 (liked) or 0 (unliked).\n"
        "Label:"
    )


class HuggingFaceLlamaGenerator:
    def __init__(self, model, device_map="auto", max_new_tokens=2, quantization="none"):
        self.model = model
        self.device_map = device_map
        # Some tokenizers emit a leading whitespace token, so 1 token can be empty after strip.
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self._pipe = None
        self._pad_token_id = None
        self._generation_config = None
        self.feature_names = None

    def set_feature_names(self, feature_names):
        self.feature_names = list(feature_names)
        return self

    def fit(self, X, y):
        # Inference-only baseline, kept for sklearn-like compatibility.
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        q = (self.quantization or "none").lower()
        if self.quantization == "4bit":
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map=self.device_map,
                load_in_4bit=True,
            )
        elif self.quantization == "8bit":
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map=self.device_map,
                load_in_8bit=True,
            )
        elif q == "none":
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map=self.device_map,
            )
        else:
            raise ValueError("Invalid hf_quantization. Use one of: none, 8bit, 4bit")
        self._pad_token_id = tokenizer.pad_token_id
        self._pipe = pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            device_map=self.device_map,
        )
        # Force a clean deterministic generation config to avoid warnings
        # from checkpoints that embed sampling-only params (e.g. top_p, temperature).
        self._generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return self

    def _build_prompt(self, row):
        return _build_binary_feedback_prompt(row, self.feature_names)
    
    def predict(self, X):
        prompts = [self._build_prompt(row) for row in X]
        out = self._pipe(
            prompts,
            generation_config=self._generation_config,
            return_full_text=False,
            batch_size=16,
        )
        preds = []
        for p in out:
            text = p[0]["generated_text"].strip()
            preds.append(_label_to_int(text))
        return np.array(preds, dtype=int)


class HuggingFaceLlamaGeneratorFinetune:
    # compact PEFT-enabled finetune: do NOT use quantized model for training
    def __init__(self, model, epochs=1, learning_rate=2e-5, max_length=512, batch_size=1, max_new_tokens=3, quantization="none"):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.tokenizer = None
        self.model_obj = None
        self._pipe = None
        self._generation_config = None
        self.feature_names = None
        safe_name = self.model.replace("/", "__")
        self.output_dir = os.path.join("results", "hf_llama_gen_finetuned", safe_name)

    def set_feature_names(self, feature_names):
        self.feature_names = list(feature_names)
        return self

    def fit(self, X, y):
        # Use PEFT (LoRA) on a full (non-quantized) base model.
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, GenerationConfig, pipeline
        from torch.utils.data import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType

        # load tokenizer and full-precision model for training
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_exists = os.path.exists(os.path.join(self.output_dir, "adapter_config.json"))
        if adapter_exists:
            # Reuse an existing finetuned adapter if already present.
            base_model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto")
            self.model_obj = PeftModel.from_pretrained(base_model, self.output_dir)
            self._pipe = pipeline("text-generation", model=self.model_obj, tokenizer=self.tokenizer, device_map="auto")
            self._generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return self

        # IMPORTANT: do not pass load_in_4bit/load_in_8bit here
        self.model_obj = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto")
        # reduce memory: disable cache and enable gradient checkpointing
        try:
            self.model_obj.config.use_cache = False
        except Exception:
            pass
        try:
            self.model_obj.gradient_checkpointing_enable()
        except Exception:
            pass

        # attach LoRA adapters
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model_obj = get_peft_model(self.model_obj, peft_config)

        texts = [(_build_binary_feedback_prompt(r, self.feature_names) + " " + str(int(l))).strip() for r, l in zip(X, y)]
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        class _DS(Dataset):
            def __init__(self, enc):
                self.enc = enc
            def __len__(self):
                return self.enc["input_ids"].size(0)
            def __getitem__(self, i):
                item = {k: v[i] for k, v in self.enc.items()}
                item["labels"] = item["input_ids"].clone()
                return item

        trainer = Trainer(
            model=self.model_obj,
            args=TrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                logging_steps=50,
                save_strategy="no",
                report_to=[],
                # Enable mixed precision training in 16-bit.
                fp16=True,
                bf16=False,
            ),
            train_dataset=_DS(enc),
        )
        trainer.train()
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_obj.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # inference can be quantized later if desired; use pipeline from trained model
        self._pipe = pipeline("text-generation", model=self.model_obj, tokenizer=self.tokenizer, device_map="auto")
        self._generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self

    def predict(self, X):
        prompts = [_build_binary_feedback_prompt(r, self.feature_names) for r in X]
        out = self._pipe(
            prompts,
            generation_config=self._generation_config,
            return_full_text=False,
            batch_size=8,
        )
        preds = []
        for p in out:
            text = p[0]["generated_text"].strip()
            preds.append(_label_to_int(text))
        return np.array(preds, dtype=int)




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
    elif model_name == "llama_gen":
        return HuggingFaceLlamaGenerator(
            model=args.hf_model,
            device_map=args.hf_device_map,
            max_new_tokens=args.hf_max_new_tokens,
            quantization=args.hf_quantization,
        )
    elif model_name == "llama_gen_finetune":
        return HuggingFaceLlamaGeneratorFinetune(
            model=args.hf_model,
            epochs=args.hf_epochs,
            learning_rate=args.hf_learning_rate,
            max_length=args.hf_max_length,
            batch_size=args.hf_batch_size,
            max_new_tokens=args.hf_max_new_tokens,
            quantization=args.hf_quantization,
        )
    else:
        raise ValueError(
            "Invalid model name. Options: tabpfn2.5, tabpfn2.6, tabpfn3.0, finetune_tabpfn3.0, "
            "xgboost, lightgbm, catboost, randomforest, random, tabicl, "
            "llama_gen "
            "llama_gen_finetune, llama_class_finetune"
        )



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
    args.add_argument(
        "--model",
        type=str,
        default="tabpfn3.0",
        help=(
            "Comma-separated models. Supports tabpfn2.5, tabpfn2.6, tabpfn3.0, finetune_tabpfn3.0, "
            "xgboost, lightgbm, catboost, randomforest, random, tabicl, "
            "llama_gen, llama_class, "
            "llama_gen_finetune, llama_class_finetune"
        ),
    )
    args.add_argument("--policy", type=str, default="standard", help="The policy to use. Options: standard, random")
    args.add_argument("--k_core", type=int, default=5, help="The k-core to use.")
    # in case of finetuning tabpfn3.0, we want to specify the number of epochs and learning rate
    args.add_argument("--epochs", type=int, default=5, help="The number of epochs to finetune tabpfn3.0")
    args.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate to finetune tabpfn3.0")
    args.add_argument(
        "--hf_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help=(
            "Hugging Face model id. Used with llama_gen/llama-class and finetune variants. "
            "For llama_class inference/fine-tuning, use a sequence-classification-compatible model."
        ),
    )
    args.add_argument(
        "--hf_device_map",
        type=str,
        default="auto",
        help="Device map for HF generation pipeline. Used with --model llama_gen",
    )
    args.add_argument(
        "--hf_max_new_tokens",
        type=int,
        default=3,
        help="Max new tokens for generation. Used with --model llama_gen",
    )
    args.add_argument(
        "--hf_quantization",
        type=str,
        default="4bit",
        choices=["none", "8bit", "4bit"],
        help="Quantization mode for HF models. Use 4bit for lowest VRAM.",
    )
    args.add_argument("--hf_epochs", type=int, default=5, help="Epochs for llama_*_finetune")
    args.add_argument("--hf_learning_rate", type=float, default=2e-5, help="Learning rate for llama_*_finetune")
    args.add_argument("--hf_max_length", type=int, default=1024, help="Max token length for llama_class and llama_*_finetune")
    args.add_argument("--hf_batch_size", type=int, default=8, help="Batch size for llama_class and llama_*_finetune")
    args.add_argument("--device", type=str, default="cuda:1", help="Device to use, e.g. cpu or cuda:1")
    # explainability
    args.add_argument("--explain", type=bool, default=False, help="Whether to compute SHAP values for tabpfn models (only for tabpfn2.5, tabpfn2.6 and tabpfn3.0)")
    args = args.parse_args()

    # Configure CUDA device without using CUDA_VISIBLE_DEVICES
    try:
        import torch
        if isinstance(args.device, str) and args.device.startswith("cuda:"):
            idx = int(args.device.split(":", 1)[1])
            torch.cuda.set_device(idx)
        # Mirror hf_device_map to keep backward compatibility with existing code
        args.hf_device_map = args.device
    except Exception:
        pass

    setproctitle.setproctitle(f"Predicting explicit negative feedback in short-video recsys ")

    dataset = get_dataset(args)
    
    track_time = defaultdict(dict)

    for model_name in args.model.split(","):
        print(f"Training model: {model_name}")
        model = get_model(args, model_name)

        # Split the dataset into train, validation, and test sets, grouped by user id
        train_df, test_df = split_group_by_user(dataset)

        print("Train set size:", len(train_df))
        print("Test set size:", len(test_df))

        feature_names = [c for c in train_df.columns if c != "label"]
        if hasattr(model, "set_feature_names"):
            model.set_feature_names(feature_names)

        # Create chunks first, then merge into single arrays for model APIs.
        train_data, train_labels = vectorize_data(train_df)
        test_data, test_labels = vectorize_data(test_df)

        # Training
        data = (train_data, train_labels)

        model, time = train(model, data)

        if model_name in ['llama_gen_finetune']:
            model_name="Llama-3.2-1B-Instruct_PEFT"
        elif model_name in ['llama_gen']:
            model_name="Llama-3.2-1B-Instruct_zero-shot"


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
            # Keep feature names before vectorization (numpy arrays do not have .columns).
            feature_names = [c for c in test_df.columns if c != "label"]

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

        
