import copy
import logging
import os
import sys
import utils
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset

from args import IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_max_len: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.pad_to_max_len:
            input_ids.append(
                torch.zeros([self.tokenizer.model_max_length],
                            dtype=input_ids[0].dtype))
            labels.append(
                torch.zeros([self.tokenizer.model_max_length],
                            dtype=labels[0].dtype))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        if self.pad_to_max_len:
            input_ids = input_ids[:-1]
            labels = labels[:-1]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class SupervisedRawDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedRawDataset, self).__init__()

        targets = [f"{target}{tokenizer.eos_token}" for target in targets]
        
        self.examples = [s + t for s, t in zip(sources, targets)]
        print(f"load {len(self.examples)} samples.....")
        self.sources = sources
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        examples_tokenized = self.tokenizer(
            self.examples[i],
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        sources_tokenized = self.tokenizer(
            self.sources[i],
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = examples_tokenized["input_ids"].squeeze()
        label = copy.deepcopy(input_ids)
        source_len = sources_tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        label[:source_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=label)


class InputOutputDataset(Dataset):
    def __init__(self, sources: Sequence[str],
                       targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_source_length: int,
                 max_target_length: int):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> dict:
        source_item = self.sources[i]
        target_item = self.targets[i]

        a_ids = self.tokenizer.encode(text=source_item, add_special_tokens=True, truncation=True,
                                      max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=target_item, add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def get_raw_data(tokenizer: transformers.PreTrainedTokenizer,
                 data_path,
                 eval_num=0,
                 model_name=""):
    logging.warning("Loading data...")

    train_file = os.path.join(data_path, "train.json")
    dev_file = os.path.join(data_path, "dev.json")
    
    if os.path.exists(train_file):
        train_data = load_dataset("json", data_files=train_file)["train"]
    if os.path.exists(dev_file):
        eval_data = load_dataset("json", data_files=dev_file)["train"]
    else:
        if eval_num > 0:
            train_val = train_data["train"].train_test_split(test_size=eval_num,
                                                    shuffle=True,
                                                    seed=42)
            train_data = train_val["train"]
            eval_data = train_val["test"]
        else:
            train_data = train_data["train"].shuffle(seed=42)
            eval_data = None

    logging.warning("Formatting inputs...")
    
    train_sources = [
        f"{tokenizer.bos_token}[INST] {example['input']} [/INST]" for example in train_data
    ]
    train_targets = [
            f"{example['target']}"
            for example in train_data
        ]
        

    eval_sources, eval_targets = [], []
    if eval_data is not None:
        eval_sources = [
            f"{tokenizer.bos_token}[INST] {example['input']} [/INST]" for example in eval_data
        ]
        eval_targets = [
            f"{example['target']}"
            for example in eval_data
        ]

    return train_sources, train_targets, eval_sources, eval_targets


def make_train_eval_dataset(tokenizer: transformers.PreTrainedTokenizer,
                            data_path,
                            training_args,
                            eval_num=0):
    print(f"load data from data_path")
    train_sources, train_targets, eval_sources, eval_targets = get_raw_data(
        tokenizer, data_path, eval_num, training_args.model_name)
    if "chatglm" in training_args.model_name:
        return InputOutputDataset(
            train_sources, train_targets, tokenizer, 2048, 512), \
            InputOutputDataset(
                eval_sources, eval_targets, tokenizer, 2048, 512)
    else:
        return SupervisedRawDataset(train_sources, train_targets, tokenizer), SupervisedRawDataset(eval_sources, eval_targets, tokenizer) if eval_num > 0 else None

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args, model=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset, eval_dataset = make_train_eval_dataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        training_args=training_args,
        eval_num=data_args.eval_num)

    print(f"using pad_to_max_len {training_args.pad_to_max_len}")
    if "chatglm" in training_args.model_name:
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )
    else:
        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, pad_to_max_len=training_args.pad_to_max_len)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
