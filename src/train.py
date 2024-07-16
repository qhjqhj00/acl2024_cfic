#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import math
import numpy as np
import torch

from transformers import Trainer as TrainerHf

from data import make_supervised_data_module
import transformers
from peft import get_peft_model, LoraConfig
from peft_lora_utils import SavePeftModelCallback, SavePeftModelAtEndCallback, resume_lora_model_from_checkpoint
from args import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_config_path: Optional[str] = field(default=None)
    model_size: Optional[str] = field(default="")  # deprecated, don't use it
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_num: int = field(default=0, metadata={"help": "Num of eval samples."})


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel, model_args: ModelArguments):

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LlamaTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    # setup scale
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # setup tokenizer
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # setup tokenizer

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    assert tokenizer.bos_token is not None
    assert tokenizer.eos_token is not None
    assert tokenizer.unk_token is not None

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # setup model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")

    # torch.set_default_tensor_type(torch.FloatTensor)

    print(f'using model arch {type(model)}')

    if training_args.use_lora:

        lora_target_modules = [
            "q_proj",
            "k_proj",
            "o_proj",
            "v_proj",
        ]
        config = LoraConfig(
            r=training_args.lora_dim,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        model.print_trainable_parameters()
        if training_args.lora_model_path:
            resume_lora_model_from_checkpoint(model,
                                              training_args.lora_model_path)

    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        model_args=model_args)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # setup dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              training_args=training_args)
    # train
    trainer = TrainerHf(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SavePeftModelCallback, SavePeftModelAtEndCallback]
        if training_args.use_lora else None,
        **data_module)
    trainer.train()


if __name__ == "__main__":
    train()
