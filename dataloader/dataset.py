from datasets import Dataset
import json
import pandas as pd

class PaperDataset:
    @staticmethod
    def format_data(dataset_path):
        """加载并格式化论文摘要-标题数据"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df["text"] = df.apply(lambda x: 
        "[INSTRUCTION] Generate a concise academic paper title based on the abstract.\n"
        f"[ABSTRACT] {x['abstract']}\n"
        f"[TITLE] {x['title']}</s>",  # 添加结束符
        axis=1)
        
        dataset = Dataset.from_pandas(df)
        return dataset.train_test_split(seed=42, test_size=0.1)
