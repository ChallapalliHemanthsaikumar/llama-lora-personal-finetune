# LLaMA LoRA Fine-Tuning Project

Welcome! This documentation will guide you through fine-tuning LLaMA using **LoRA** on a personal dataset.

## Sections
- [Dataset Preparation](dataset.md)
- [LoRA Configuration](lora_config.md)
- [Training the Model](training.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
- [Resources & Notebooks](resources.md)




## Colab Notebook

You can run the full LoRA fine-tuning workflow in Google Colab:




##### Python packages 

!pip install --upgrade transformers accelerate bitsandbytes
!pip install bert-score sentence-transformers matplotlib
!pip install rouge_score evaluate nltk
 
# Optional


##### Optional W&B login:

import wandb
wandb.login()  # Enter your W&B token




Hardware / Runtime

Google Colab GPU (A100/V100 recommended)

Minimum 12–16GB RAM 

Here I used T4 GPU 





[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/githubChallapalliHemanthsaikumar/llama-lora-personal-finetune/blob/main/notebooks/LLaMA_LoRA_Personal_Finetune.ipynb)







See the [Training Guide](training.md) for detailed steps.

