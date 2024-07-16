from collections import defaultdict
import torch
from nltk import sent_tokenize
from torch.nn.functional import softmax

def tokenize_sentences(sentences, tokenizer):
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentences]
    token_ids_list = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    return tokenized_sents, token_ids_list

def get_first_token_ids(tokenized_sents, token_ids_list, tokenizer, token_position=0):
    first_token_ids = defaultdict(list)
    first_token_ids_set = set()
    
    for i, sent in enumerate(tokenized_sents):
        if token_position < len(sent)-1:
            token_ids = token_ids_list[i][token_position]
            first_token_ids[i].append(token_ids)
            first_token_ids_set.add(token_ids)

            tmp_token = sent[token_position]
            if tmp_token.startswith("▁") and tmp_token != "▁":
                tmp_token = tmp_token[1:]
                token_ids = tokenizer.convert_tokens_to_ids(tmp_token)
                first_token_ids[i].append(token_ids)
                first_token_ids_set.add(token_ids)
                
    return first_token_ids, first_token_ids_set


def find_top_k_sentences(sentences, tokenizer, logits, k=1, token_position=0):
    tokenized_sents, token_ids_list = tokenize_sentences(sentences, tokenizer)
    first_token_ids, first_token_ids_set = get_first_token_ids(tokenized_sents, token_ids_list, tokenizer, token_position)

    mask = torch.ones_like(logits).scatter_(0, torch.tensor(list(first_token_ids_set)), 0)
    masked_logits = logits.masked_fill(mask.bool(), float('-inf'))
    probabilities = softmax(masked_logits, dim=0)
    
    topk_prob_values, topk_token_ids = torch.topk(probabilities, k)
    
    return get_sentences_from_ids(sentences, first_token_ids, topk_token_ids, topk_prob_values)

def get_sentences_from_ids(sentences, first_token_ids, topk_token_ids, topk_prob_values):
    top_sentences = {}
    for sent_idx, token_ids in first_token_ids.items():
        for token_id in token_ids:
            if token_id in topk_token_ids:
                token_pos = (topk_token_ids == token_id).nonzero(as_tuple=True)[0]
                prob_value = topk_prob_values[token_pos].item()
                sentence = sentences[sent_idx]
                
                if token_id not in top_sentences:
                    top_sentences[token_id] = []
                top_sentences[token_id].append((sentence, prob_value))
    
    return top_sentences

def get_single_sent(prompt_kv_cache, top_sent, token_id, tokenizer, model, position=1):
    suffix = ""
    while len(top_sent) > 1:
        suffix += tokenizer.decode([token_id])
        inputs = tokenizer(suffix, return_tensors="pt").to(model.device)
        inputs['past_key_values'] = prompt_kv_cache
        with torch.no_grad():
            logits = model(**inputs, use_cache=True).logits
            
        sents = [s[0] for s in top_sent]
        prob = softmax(logits[0,-1,:], dim=-1)
        top_sents = find_top_k_sentences(sents, tokenizer, prob.cpu(), 1, position)
        top_sent = list(top_sents.values())[0]
        token_id = list(top_sents.keys())[0]
        position += 1
    return top_sent

def expand_sent(sent, all_sents, expand_size):
    length = len(sent.split())
    sent_idx = all_sents.index(sent)
    rtn = [sent]
    for tmp_sent in all_sents[sent_idx+1:]:
        tmp_length = len(tmp_sent.split())
        if length+tmp_length > expand_size:
            break
        else:
            rtn.append(tmp_sent)
            length+=tmp_length
    return rtn
    
def constrained_prefix(prompt_kv_cache, input_text, prompt_logits, tokenizer, model, expand_size=0, topk=3):
    rtn = []
    all_sents = sent_tokenize(input_text)[3:-3] 
    prob = softmax(prompt_logits[0,-1,:], dim=-1)
    top_sents = find_top_k_sentences(all_sents, tokenizer, prob.cpu(), topk)
    for token_id in top_sents:
        single_sent = get_single_sent(prompt_kv_cache, top_sents[token_id], token_id, tokenizer, model)[0][0]
        rtn.append(single_sent)
    rtn = list(set(rtn))

    return rtn, all_sents

def clone_key_value_cache(key_value_cache):
    if key_value_cache is None:
        return None
    
    return [
        tuple(layer_cache.clone() for layer_cache in layer)
        for layer in key_value_cache
    ]

def skip_decoding(kv_cache, sent_prefixes, all_sents, tokenizer, model, max_length=512):
    rtn = []
    
    for prefix in sent_prefixes:
        tmp_kv_cache = clone_key_value_cache(kv_cache)
        candidate_sents = expand_sent(prefix, all_sents, max_length)
        
        eos_probs = []
        for i in range(len(candidate_sents)):
            if i == 0:
                surfix = candidate_sents[i]
            else:
                surfix = " " + candidate_sents[i]
            inputs = tokenizer(surfix, return_tensors="pt").to(model.device)
            inputs['past_key_values'] = tmp_kv_cache
            with torch.no_grad():
                outs = model(**inputs, use_cache=True)
                logits = outs.logits
                tmp_kv_cache = outs.past_key_values
                last_token_prob = softmax(logits[0,-1,:], dim=-1)
                eos_prob = last_token_prob[tokenizer.eos_token_id]
                eos_probs.append(eos_prob.cpu().tolist())
        max_index = eos_probs.index(max(eos_probs))
        out_sents = candidate_sents[:max_index+1]
        out_sents_idx = [all_sents.index(sent) for sent in out_sents]
        rtn.append(out_sents_idx)
            
    return rtn 

def merge_overlapping_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged_intervals = [intervals[0]]

    for current in intervals:
        last = merged_intervals[-1]

        if current[0] <= last[-1]:
            merged_intervals[-1] = list(range(last[0], max(last[-1], current[-1]) + 1))
        else:
            merged_intervals.append(current)

    return merged_intervals

def sample_sents(input_text, tokenizer, model, max_length=512, topk=3):
    prompt = f"{tokenizer.bos_token}[INST] {input_text} [/INST]"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model_outputs = model(**prompt_inputs)
        prompt_kv_cache = model_outputs.past_key_values
        prompt_logits = model_outputs.logits
        
    sent_prefixes, all_sents = constrained_prefix(prompt_kv_cache, input_text, prompt_logits, tokenizer, model, topk)
    all_spans_idx = skip_decoding(prompt_kv_cache, sent_prefixes, all_sents, tokenizer, model, max_length)
    all_spans_idx = merge_overlapping_intervals(all_spans_idx)
    all_spans = []
    
    for span_idx in all_spans_idx:
        tmp = []
        for idx in span_idx:
            tmp.append(all_sents[idx])
        all_spans.append(" ".join(tmp))
        
    return all_spans
    
if __name__ == "__main__":
    import transformers
    import json
    model = transformers.AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.bfloat16,
            cache_dir = "/share/shared_models",
            attn_implementation="flash_attention_2")
    model.cuda()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir = "/share/shared_models")

    eval_data = json.load(open("data/longbench/qasper.json"))
    ## kv_cache
    import time

    for i in range(10):
        start = time.time()
        res = sample_sents(eval_data[i]["input"], tokenizer, model, 128)
        print(time.time()-start)
        print("+++"*20, "\n")


