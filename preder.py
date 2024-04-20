from transformers import Seq2SeqTrainer
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


def predict(
    trainer: Seq2SeqTrainer,
    data_args,
    model_args,
    training_args,
    predict_dataset,
    tokenizer,
):
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 进行预测
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
        )

        # 记录预测指标
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # 保存预测结果
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(
                    predictions != -100, predictions, tokenizer.pad_token_id
                )
                predictions = tokenizer.batch_decode(
                    predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))
