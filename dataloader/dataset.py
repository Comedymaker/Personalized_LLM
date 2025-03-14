from datasets import Dataset
import json
import pandas as pd
from models.tokenizer import Tokenizer

class PaperDataset:
    @staticmethod
    def format_data(dataset_path):
        """加载并格式化论文摘要-标题数据"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        # df["text"] = df.apply(lambda x: 
        # "[INSTRUCTION] Generate a concise academic paper title based on the abstract.\n"
        # f"[ABSTRACT] {x['abstract']}\n"
        # f"[TITLE] {x['title']}</s>",  # 添加结束符
        # axis=1)
         # 格式化为对话格式，前面加上提示语
        # df["text"] = df[["abstract", "title"]].apply(lambda x: 
        #     "<|im_start|>user\n" + x["abstract"] + " <|im_end|>\n<|im_start|>assistant\n" + x["title"] + "<|im_end|>\n", axis=1)

        df["text"] = df[["abstract", "title"]].apply(lambda x: 
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "Generate a concise academic paper title based on the abstract",
            },
            {"role": "user", "content": x["abstract"]},
            {"role": "assistant", "content": x["title"]}
        ], tokenize=False, add_generation_prompt=False), axis=1)

        dataset = Dataset.from_pandas(df)

        return dataset.train_test_split(seed=42, test_size=0.1)

    def format_data_combModel(dataset_path):
        """加载并格式化论文摘要-标题数据"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        tokenizer = Tokenizer.load_tokenizer()

        # df["text"] = df.apply(lambda x: 
        # "[INSTRUCTION] Generate a concise academic paper title based on the abstract.\n"
        # f"[ABSTRACT] {x['abstract']}\n"
        # f"[TITLE] {x['title']}</s>",  # 添加结束符
        # axis=1)
         # 格式化为对话格式，前面加上提示语
        # df["text"] = df[["abstract", "title"]].apply(lambda x: 
        #     "<|im_start|>user\n" + x["abstract"] + " <|im_end|>\n<|im_start|>assistant\n" + x["title"] + "<|im_end|>\n", axis=1)

        df["text"] = df[["abstract", "title"]].apply(lambda x: 
            tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "Generate a concise academic paper title based on the abstract",
            },
            {"role": "user", "content": x["abstract"]},
        ], tokenize=False, add_generation_prompt=True), axis=1)

        dataset = Dataset.from_pandas(df)
        
        return dataset.train_test_split(seed=42, test_size=0.1)

    def preprocess_function(examples):
        inputs = ["Generate a concise academic paper title based on the abstract: " + doc for doc in examples["abstract"]]  # 输入文本
        targets = examples["title"]     # 目标文本（标签）

        tokenizer = Tokenizer.load_tokenizer()

        # 将输入文本tokenize
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

        # 将目标文本tokenize（用于生成任务）
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)

        model_inputs["labels"] = labels["input_ids"]  # 把目标标签放到model_inputs中
        return model_inputs

