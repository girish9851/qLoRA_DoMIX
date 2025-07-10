#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# DoMIX for Question-Answering (QA) in Colab
# ------------------------------------------------
# This script demonstrates a modular DoMIX-style domain adaptation setup
# using a QLoRA approach for QA datasets. It uses TinyLlama for efficiency.
# Works in Colab without gated models or local caching issues.

# -----------------------------------------------
# SECTION 1: Install & Import Dependencies
# -----------------------------------------------
get_ipython().system('pip install -q transformers datasets accelerate peft bitsandbytes')

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# -----------------------------------------------
# SECTION 2: Load and Prepare Dataset (MedMCQA)
# -----------------------------------------------
# Load a small subset for demonstration
qa_dataset = load_dataset("openlifescienceai/medmcqa", split="train[:1000]")

# Format each example into a prompt+answer style
def format_mcqa(example):
    question = example["question"].strip()
    options = [example[k].strip() for k in ["opa", "opb", "opc", "opd"]]
    choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    correct = chr(65 + int(example["cop"]))  # e.g. 'C'
    return {
        "text": f"User: {question}\n{choices}\nAssistant: The correct answer is {correct}."
    }

formatted = qa_dataset.map(format_mcqa, remove_columns=qa_dataset.column_names)


# In[3]:


# -----------------------------------------------
# SECTION 3: Tokenization
# -----------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = formatted.map(tokenize, batched=True, remove_columns=formatted.column_names)



# In[4]:


# -----------------------------------------------
# SECTION 4: Load Model with QLoRA & DoMIX Concept
# -----------------------------------------------
# Use 4-bit quantized TinyLlama and add domain adapter
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map={"": 0},  # force loading on cuda:0 only
    trust_remote_code=True
)

# Prepare for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


# In[5]:


# -----------------------------------------------
# SECTION 5: Training Setup
# -----------------------------------------------
training_args = TrainingArguments(
    output_dir="./qa_domix_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    report_to=[],
    run_name="domix_qa_finetune"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)


# In[6]:


# -----------------------------------------------
# SECTION 6: Train & Save Adapter
# -----------------------------------------------
trainer.train()
model.save_pretrained("./qa_domix_adapter")
tokenizer.save_pretrained("./qa_domix_adapter")

print("✅ QLoRA QA fine-tuning with DoMIX-style modularization complete.")


# In[7]:


def answer_question(question):
    prompt = f"User: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
print("\n💬 Inference Example:")
print(answer_question(
    "Which medicine should I take when I have fracture ?"
))


# In[8]:


from peft import PeftModel, prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model #, merge_adapter

# -------------------------------------------
# SECTION 2: Load Fine-Tuned Model + Adapter
# -------------------------------------------
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "./qa_domix_adapter")

# Merge adapter weights into the base model for inference
model = model.merge_and_unload()
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./qa_domix_adapter")
tokenizer.pad_token = tokenizer.eos_token



# In[9]:


# -------------------------------------------
# SECTION 3: Evaluation Function on Test QA
# -------------------------------------------
# Load test set (same domain)
test_dataset = load_dataset("openlifescienceai/medmcqa", split="validation[:200]")

def format_qa(example):
    question = example["question"].strip()
    options = [example[k].strip() for k in ["opa", "opb", "opc", "opd"]]
    choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    correct = chr(65 + int(example["cop"]))  # e.g. 'C'
    return {
        "input": f"User: {question}\n{choices}\nAssistant:",
        "expected": correct
    }

test_data = test_dataset.map(format_qa)

correct = 0
for sample in test_data:
    inputs = tokenizer(sample["input"], return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Assistant:")[-1].strip()
    print(answer)
    if sample["expected"] in answer:
        correct += 1

accuracy = correct / len(test_data)
print(f"✅ QA Accuracy on MedMCQA validation split: {accuracy:.2%}")


# In[10]:


# DoMIX for Question-Answering (QA) in Colab with Modular Adapter Routing & Fusion
# ------------------------------------------------------------------------
# This script demonstrates domain adaptation using LoRA for different QA domains
# Includes: router-based adapter selection, merge-and-unload logic, and AdapterFusion skeleton.

# !pip install -q transformers datasets accelerate peft

# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM,
#     TrainingArguments, Trainer,
#     DataCollatorForLanguageModeling
# )
from peft import (
    prepare_model_for_kbit_training, LoraConfig, get_peft_model,
    PeftModel, PeftConfig
)
import os

# --------------------------
# SECTION 1: Helper Functions
# --------------------------
def format_qa(example):
    question = example["question"].strip()
    options = [example[k].strip() for k in ["opa", "opb", "opc", "opd"]]
    choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    correct = chr(65 + int(example["cop"]))
    return {
        "text": f"User: {question}\n{choices}\nAssistant: The correct answer is {correct}."
    }

def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

def load_tokenized_dataset(tokenizer, domain="medical"):
    if domain == "medical":
        dataset = load_dataset("openlifescienceai/medmcqa", split="train[:1000]")
        dataset = dataset.map(format_qa)
    elif domain == "legal":
        dataset = load_dataset("lex_glue", "ledgar", split="train[:1000]")
        def format_legal(example):
            return {"text": f"User: {example['text']}\\nAssistant: {example['label']}"}
        dataset = dataset.map(format_legal)
    elif domain == "finance":
        dataset = load_dataset("financial_phrasebank", split="train[:1000]")
        def format_finance(example):
            return {"text": f"User: {example['sentence']}\\nAssistant: {example['label']}"}
        dataset = dataset.map(format_finance)
    else:
        raise ValueError("Unsupported domain")

    return dataset.map(lambda ex: tokenize(ex, tokenizer), batched=True, remove_columns=dataset.column_names)

# --------------------------
# SECTION 2: Load Base Model
# --------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,
    device_map={"": 0},  # force single GPU
    trust_remote_code=True
)

# --------------------------
# SECTION 3: Train and Save Adapters
# --------------------------
def train_and_save_adapter(domain_name, dataset, tokenizer):
    model = prepare_model_for_kbit_training(base_model)
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=f"./adapter_{domain_name}",
        run_name=f"lora_{domain_name}",
        report_to=[],
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(f"./adapter_{domain_name}")
    tokenizer.save_pretrained(f"./adapter_{domain_name}")
    return f"./adapter_{domain_name}"

# --------------------------
# SECTION 4: Adapter Router
# --------------------------
def router(domain):
    if domain == "medical":
        return "./adapter_medical"
    elif domain == "legal":
        return "./adapter_legal"
    elif domain == "finance":
        return "./adapter_finance"
    else:
        raise ValueError("Unknown domain")

# --------------------------
# SECTION 5: Load Adapter Dynamically
# --------------------------
def load_model_with_adapter(base_model_name, adapter_path):
    config = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # load_in_8bit=True,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model

# --------------------------
# SECTION 6: Merge Adapters (Optional)
# --------------------------
def merge_adapter(model):
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./merged_model")
    return merged_model

# --------------------------
# SECTION 7: Main Flow
# --------------------------
if __name__ == "__main__":
    # Train medical adapter if not exists
    if not os.path.exists("./adapter_medical"):
        tokenized_dataset = load_tokenized_dataset(tokenizer, domain="medical")
        train_and_save_adapter("medical", tokenized_dataset, tokenizer)

    # Router selects adapter path
    adapter_path = router("medical")
    model = load_model_with_adapter(model_name, adapter_path)

    # Optionally merge adapter
    # model = merge_adapter(model)

    print("✅ Model loaded with domain-specific adapter")

