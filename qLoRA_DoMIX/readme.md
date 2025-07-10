# 🔌 Plug-and-Play LLMs: Fine-Tuning with DoMIX and QLoRA

---

## 📄 Paper Link  
**DoMIX: Domain Mixture of Experts for Parameter-Efficient Fine-Tuning**  
🔗 https://arxiv.org/abs/2507.02302

---

## ⚙️ What This Project Does  
This project shows how to fine-tune a quantized LLM using **QLoRA** and manage multiple **domain-specific adapters** via the **DoMIX** approach. It supports:

- Training lightweight LoRA adapters per domain (e.g. medical, finance)
- Dynamically routing user queries to the correct adapter
- Optional merging of adapters into the base model for deployment
- Full Colab compatibility and low-VRAM support

---

## 💡 Use Cases  
- Domain-aware chatbots (e.g. healthcare assistant, financial QA)
- Modular fine-tuning pipelines for multi-tenant LLM services
- Shipping adapters independently of base models

---

## 🔗 References  
- [DoMIX Paper](https://arxiv.org/abs/2507.02302)  
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)  
- [PEFT Library (Hugging Face)](https://github.com/huggingface/peft)  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

