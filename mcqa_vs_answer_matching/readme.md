Answer Matching vs MCQ: A Practical LLM Evaluation Toolkit

Link:https://arxiv.org/pdf/2507.02856 

contents: a prototype experiment using

GEN_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"   # For generation
MATCH_MODEL = "Qwen/Qwen1.5-0.5B" for matching 

This repository implements a simplified version of the paper “Answer Matching Outperforms Multiple Choice for Language Model Evaluation” ( 2025). We compare two evaluation strategies—multiple choice (MCQ) and free-form answer matching—using openly available small language models. The toolkit includes a sample dataset of 20 elementary-level questions with sentence-form reference answers, along with generation and matching pipelines.

While MCQs are easy to grade, they often let models guess the correct answer using patterns rather than actual reasoning. Answer matching, on the other hand, evaluates generative ability more accurately by comparing model outputs to reference answers using a second LLM as a matcher. This repo demonstrates how even small models can be used for this purpose efficiently and reliably.
