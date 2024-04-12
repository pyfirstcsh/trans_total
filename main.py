# main.py
import os
import sys
import logging
import datasets
import transformers

from arguments import parse_arguments
from datapre import load_and_preprocess_data
from model import load_model_and_tokenizer
from trainer import initialize_trainer, train_and_evaluate
from preder import predict
from transformers import set_seed, DataCollatorForSeq2Seq

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:  # noqa: F841
#     pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 设置日志

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 解析命令行参数
    model_args, data_args, training_args = parse_arguments()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # 设置随机种子，确保可重复性
    set_seed(training_args.seed)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args)
    # 加载和预处理数据集
    train_dataset, eval_dataset, predict_dataset = load_and_preprocess_data(
        data_args, model_args, training_args, tokenizer
    )

    # 初始化数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # 初始化训练器
    trainer = initialize_trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练和评估
    train_and_evaluate(trainer, training_args, train_dataset, eval_dataset, data_args)

    # 预测
    predict(trainer, data_args, model_args, training_args, predict_dataset, tokenizer)
    logger.info("Training, evaluation, and prediction have finished")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [
        l
        for l in [data_args.source_lang, data_args.target_lang]  # noqa: E741
        if l is not None
    ]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
