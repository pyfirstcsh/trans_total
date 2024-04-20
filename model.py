# model.py
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


def load_model_and_tokenizer(model_args):
    # 加载配置
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
