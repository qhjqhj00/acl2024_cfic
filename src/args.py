# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from packaging import version

from transformers.utils import (
    ExplicitEnum,
    cached_property,
    ccl_version,
    get_full_repo_name,
    is_accelerate_available,
    is_psutil_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_neuroncore_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    logging,
    requires_backends,
)
import transformers
from transformers.training_args import *

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class LlamaTrainingArguments(transformers.TrainingArguments):
    """
    LlamaTrainingArguments inherits from transformers.TrainingArguments

    """
    master_addr: str = field(
        default="", metadata={"help": "Master address: for distribuetd training."}
    )
    master_port: int = field(
        default=-1, metadata={"help": "Master port: for distribuetd training."}
    )
    cache_dir: Optional[str] = field(default=None)
    
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    save_on_each_node: bool = field(
        default=False
    )
    global_rank: int = field(
        default=-1, metadata={"help": "Global rank: for distributed training."}
    )
    node_world_size: int = field(
        default=0, metadata={"help": "World size: for distributed training."}
    )
    pp_num_stages: int = field(default=2, metadata={"help": "num of pp stages"})
    dp_world_size: int = field(default=2, metadata={"help": "size of dp world"})
    dp_rank: int = field(default=-1, metadata={"help": "rank in dp world"})

    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    pad_to_max_len: bool = field(default=False, metadata={"help": "whether to pad to max length"})
    pretokenized_train_data_path: str = field(
        default="", metadata={"help": "pretokenized_train_data_path."}
    )
    pretokenized_eval_data_path: str = field(
        default="", metadata={"help": "pretokenized_eval_data_path."}
    )
    # Lora Argument
    use_lora: bool = field(default=False, metadata={"help": "whether to use lora layer"})
    lora_model_path: Optional[str] = field(default=None, metadata={"help": "specify the path to lora weight"})
    lora_dim: int = field(default=8, metadata={"help": "dimension of lora layer"})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter for Lora scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for Lora layers"})
    
    # Baichuan position embedding
    pos_embed: str = field(
        default="RoPE", metadata={"help": "Master address: for distribuetd training."}
    )

    model_name: str = field(
        default="", metadata={"help": "pretokenized_eval_data_path."}
    )

    @cached_property
    def _setup_devices(self) -> "torch.device":
        ###### add codes
        if self.deepspeed and self.node_world_size > 1:
            print(f"before configuring ds {self.master_addr} {self.master_port} ")
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            real_world_size = comm.Get_size()
            proc_pre_node = real_world_size // self.node_world_size
            self.node_world_size = real_world_size
            
            rank = comm.Get_rank()
            self.global_rank = rank            
            self.local_rank = rank % proc_pre_node
            os.environ["RANK"] = str(self.global_rank)
            os.environ["WORLD_SIZE"] = str(self.node_world_size)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            print(f"after configuring ds {os.environ['RANK']} {os.environ['WORLD_SIZE']} {os.environ['LOCAL_RANK']} {os.environ['MASTER_ADDR']} {os.environ['MASTER_PORT']}")
        ###### 
        return super()._setup_devices
