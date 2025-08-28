## Dataset Preparation

The dataset for this project was prepared by using ChatGPT to generate user questions such as "Who am I?" and similar queries. Each sample was formatted in JSONL, where each line contains an instruction and a corresponding response. For example:

```json
{"instruction": "[INST] How did Hemanth Sai Kumar improve customer support efficiency? [/INST]", "response": "He reduced customer support response times by 40% through the development of LangChain-powered AI agents."}
```

Following this approach, at least 100 such samples were created and saved as a `.jsonl` file in the repository. This dataset serves as the foundation for fine-tuning the model on personalized instructions and responses.

After creating the dataset, it was uploaded to Google Drive to enable easy access from Google Colab for training and experimentation.

