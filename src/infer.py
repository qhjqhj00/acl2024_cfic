
import os
import datasets
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from typing import Optional
from cfic_decoding import sample_sents

@dataclass
class InferArguments:
    eval_file: Optional[str] = field(default=None)
    raw_file: Optional[str] = field(default=None)
    checkpoint_path: Optional[str] = field(default=None)
    save_path: Optional[str] = field(default=None)
    batch_size: Optional[int] = field(default=4)
    max_length: Optional[int] = field(default=2048)
    max_new_token: Optional[int] = field(default=128)
    min_new_token: Optional[int] = field(default=128)
    flash_attn: Optional[bool] = field(default=False)
    n_sample: Optional[int] = field(default=1)
    limit: Optional[int] = field(default=-1)
    sample_sents: Optional[bool] = field(default=False)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_sample(
    rank, world_size, data, device, model_path, out_path, \
    max_length=32768, max_gen=512, min_gen=128, n_sample=1):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    for json_obj in tqdm(data):
        pred = sample_sents(json_obj['input'], tokenizer, model, 256, n_sample)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({f"pred": pred, "_id": json_obj["_id"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def get_pred(
    rank, world_size, data, device, model_path, out_path, \
    max_length=32768, max_gen=512, min_gen=128, n_sample=1):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    for json_obj in tqdm(data):
        prompt = f"{tokenizer.bos_token}[INST] {json_obj['input']} [/INST]"
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)


        context_length = input.input_ids.shape[-1]
        
        if n_sample == 1:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                min_new_tokens=min_gen,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            outputs = model.generate(
                **input, 
                max_new_tokens=max_gen, 
                min_new_tokens=min_gen,
                pad_token_id=tokenizer.pad_token_id,
                top_k=5,
                num_return_sequences=n_sample,
                do_sample=True,
                use_cache=True,)
            outputs = outputs[:,context_length:]
            pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({f"pred": pred, "_id": json_obj["_id"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def load_model_and_tokenizer(path, device):
    if "chatglm" in path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in path:
        replace_llama_attn_with_flash_attn()
        from fastchat.model import load_model
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in path:
        model = MistralForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2")
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    parser = HfArgumentParser(InferArguments)
    args, = parser.parse_args_into_dataclasses()
    print(args)

    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = datasets.load_dataset("json", data_files=args.eval_file, split=f"train")

    data_all = [data_sample for data_sample in data]
    
    data_subsets = [data_all[i::world_size] for i in range(world_size)]
    target_function = get_sample if args.sample_sents else get_pred

    # Create and start processes using list comprehension
    processes = [
        mp.Process(target=target_function, args=(
            rank, world_size, data_subsets[rank],
            device, args.checkpoint_path,
            args.save_path,
            args.max_length,
            args.max_new_token,
            args.min_new_token,
            args.n_sample
        )) for rank in range(world_size)
    ]

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()      
