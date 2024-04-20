# data.py
import logging

# from datasets import load_dataset # 联网使用
from datasets import load_from_disk  # 离线使用

# from sklearn.model_selection import train_test_split
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


def preprocess_function_closure(tokenizer, data_args, training_args):
    # ...
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]
    max_target_length = (
        data_args.max_target_length
        if training_args.do_train
        else data_args.val_max_target_length
    )
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples) -> BatchEncoding:
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]  # noqa: E741
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def load_and_preprocess_data(data_args, model_args, training_args, tokenizer):
    # 加载和预处理数据集的代码

    # raw_datasets = load_dataset(
    #     data_args.dataset_name,
    #     data_args.dataset_config_name,
    #     cache_dir=model_args.cache_dir,
    #     token=model_args.token,
    # )

    # # 分割数据集：90% 用于训练，剩余的 10% 将被进一步分为验证集和测试集
    # train_test_data = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)

    # # 再次分割剩余的 10% 数据为验证集和测试集
    # validation_test_data = train_test_data["test"].train_test_split(
    #     train_size=0.5, seed=20
    # )

    # # 更新 raw_datasets 字典以包含训练集、验证集和测试集
    # raw_datasets["train"] = train_test_data["train"]
    # raw_datasets["validation"] = validation_test_data["train"]
    # raw_datasets["test"] = validation_test_data["test"]

    # 离线使用,划分好了数据集
    raw_datasets = load_from_disk(data_args.dataset_name)


    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    preprocess_function = preprocess_function_closure(
        tokenizer, data_args, training_args
    )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        # max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        # eval_dataset = Dataset.from_dict(eval_dataset)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        # max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    return train_dataset, eval_dataset, predict_dataset
