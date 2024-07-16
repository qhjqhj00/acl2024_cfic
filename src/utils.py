import json
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer
from typing import List
from tqdm import tqdm

mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')

def tok(text):
    return mt.tokenize(text)

def detok(text):
    return md.detokenize(text)

def load_json(path):
    try:
        data = json.load(open(path))
    except:
        data = open(path).readlines()
        data = [json.loads(line) for line in data]
    return data

def load_jsonl(path: str) -> List:
    rtn = []
    print(f"Begin to load {path}")
    for line in tqdm(open(path)):
        line = json.loads(line)
        rtn.append(line)
    return rtn

def save_jsonl(data: list, path: str) -> None:
    with open(path, "w") as f:
        for line in data:
            f.write(
                json.dumps(
                    line, ensure_ascii=False)+"\n")

def load_json(path: str):
    with open(path, "r") as f:
        rtn = json.load(f)
    return rtn

def save_json(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
