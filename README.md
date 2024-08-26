# Chunking-Free In-Context Retrieval for RAG Systems

This repository contains the code for the paper [Grounding Language Model with Chunking-Free In-Context Retrieval](https://arxiv.org/pdf/2402.09760). The paper presents a novel approach for dynamically expanding the vocabulary of language models based on the input text, specifically tailored for Retrieval-Augmented Generation (RAG) systems. The paper is accepted by ACL 2024 main conference.

## Introduction

Traditional RAG systems often struggle with grounding responses using precise evidence text due to the challenges of processing lengthy documents and filtering out irrelevant content. Commonly employed solutions, such as document chunking and adapting language models to handle longer contexts, have their limitations. These methods either disrupt the semantic coherence of the text or fail to effectively address the issues of noise and inaccuracy in evidence retrieval.

CFIC addresses these challenges by circumventing the conventional chunking process. It utilizes the encoded hidden states of documents for in-context retrieval, employing auto-aggressive decoding to accurately identify the specific evidence text required for user queries, eliminating the need for chunking. CFIC is further enhanced by incorporating two decoding strategies, namely Constrained Sentence Prefix Decoding and Skip Decoding. These strategies not only improve the efficiency of the retrieval process but also ensure that the fidelity of the generated grounding text evidence is maintained.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/qhjqhj00/acl2024_cfic.git
cd acl2024_cfic
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run the following command:

```bash
bash scripts/train.sh
```

### Evaluation

To evaluate the model, run the following command:

```bash
bash scripts/eval_pipeline.sh
```

## Dataset

The dataset used in this project is available at [Hugging Face](https://huggingface.co/datasets/TommyChien/ACL24_CFIC/).

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{qian2024grounding,
    title = "Grounding Language Model with Chunking-Free In-Context Retrieval",
    author = "Qian, Hongjin  and
      Liu, Zheng  and
      Mao, Kelong  and
      Zhou, Yujia  and
      Dou, Zhicheng",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    pages = "1298--1311",
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact Hongjin Qian (chienqhj[at]gmail[dot]com).

```
