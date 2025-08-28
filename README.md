# llama-lora-personal-finetune


Welcome! This documentation will guide you through fine-tuning LLaMA using **LoRA** on a personal dataset.

---

## Prerequisites

Before starting, make sure you have the following:

### 1. Hugging Face Access
- Access to **gated repositories** like:
  - Meta's LLaMA 2 models (7B)
  - Google's Gemma models family
- Check your access [here](https://huggingface.co/settings/models) under **Gated Repos Status**.
- **Login in Colab / Python:**
```python
from huggingface_hub import login
login()  # Enter your Hugging Face token
