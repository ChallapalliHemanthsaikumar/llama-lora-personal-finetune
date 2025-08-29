# LLaMA LoRA Fine-Tuning Project

Welcome! This documentation will guide you through fine-tuning LLaMA using **LoRA** on a personal dataset.

## Sections
- [Setup](setup.md)
- [Dataset Preparation](dataset.md)
- [LoRA Configuration](lora_config.md)
- [Training the Model](training.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
- [Resources & Notebooks](resources.md)




## Colab Notebook

You can run the full LoRA fine-tuning workflow in Google Colab:






### Python Environment Setup

Before you begin, install the required Python packages in your Colab or local environment:

```bash
!pip install --upgrade transformers accelerate bitsandbytes
!pip install bert-score sentence-transformers matplotlib
!pip install rouge_score evaluate nltk
```

**Optional:** For experiment tracking, you can use Weights & Biases (W&B). To log in, run:

```python
import wandb
wandb.login()  # Enter your W&B token when prompted
```






Hardware / Runtime

Google Colab GPU (A100/V100 recommended)

Minimum 12–16GB RAM 

Here I used T4 GPU 





[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/githubChallapalliHemanthsaikumar/llama-lora-personal-finetune/blob/main/notebooks/LLaMA_LoRA_Personal_Finetune.ipynb)







See the [Training Guide](training.md) for detailed steps.

