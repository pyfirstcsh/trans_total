# model.py
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from model_flop import FLOPForConditionalGeneration
from model_multConv import MultConvForConditionalGeneration
from model_sparse import SparseForConditionalGeneration
from model_switch import SwitchForConditionalGeneration
from model_switch_sparse import SwitchSparseForConditionalGeneration
from model_sparse_multConv import SparseMultConvForConditionalGeneration
from model_switch_multConv import SwitchMultConvForConditionalGeneration
from config import (
    MultConvAttnConfig,
    SparseConfig,
    SwitchConfig,
    SwitchSparseConfig,
    SparseMultConvConfig,
    SwitchMultConvConfig,
)


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

    # model_name = "base"
    # model_name = "switch"
    # model_name = "sparse"
    # model_name = "flop"
    # model_name = "multConv"

    # model_name = "switch_sparse"
    # model_name = "sparse_multConv"
    model_name = "switch_multConv"

    if model_name == "base":
        # 加载模型
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "switch":
        # 创建自定义配置
        switch_config = SwitchConfig.from_dict(config.to_dict())
        model = SwitchForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=switch_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "sparse":
        # 创建自定义配置
        sparse_config = SparseConfig.from_dict(config.to_dict())
        model = SparseForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=sparse_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "flop":
        model = FLOPForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "multConv":
        # 创建自定义配置
        mult_conv_config = MultConvAttnConfig.from_dict(config.to_dict())
        model = MultConvForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=mult_conv_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "switch_sparse":
        # 创建自定义配置
        switch_sparse_config = SwitchSparseConfig.from_dict(config.to_dict())
        model = SwitchSparseForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=switch_sparse_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "sparse_multConv":
        # 创建自定义配置
        sparse_multConv_config = SparseMultConvConfig.from_dict(config.to_dict())
        model = SparseMultConvForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=sparse_multConv_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_name == "switch_multConv":
        # 创建自定义配置
        switch_multConv_config = SwitchMultConvConfig.from_dict(config.to_dict())
        model = SwitchMultConvForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=switch_multConv_config,
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
