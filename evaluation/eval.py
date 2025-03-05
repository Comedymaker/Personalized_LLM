import torch
import sys
import os
from transformers import AutoTokenizer
# 将项目根目录加入到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config_loader import load_config
from models.model import TinyModelLoader
from dataloader.dataset import PaperDataset

class Evaluator:
    def __init__(self):
        # 加载配置文件
        self.config = load_config()

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base"]["tiny_model_id"])  # 直接使用训练时的 tokenizer
        self.model = TinyModelLoader.load_finetuned_model()  # 加载微调后的模型
        self.model.eval()  # 设置模型为评估模式

        # 加载数据
        self.data = PaperDataset.format_data(self.config["base"]["dataset_path"])

    def prepare_input_for_testing(self, abstract_text):
        """
        根据训练时的格式处理输入文本
        """
        instruction = "[INSTRUCTION] Generate a concise academic paper title based on the abstract."
        abstract = f"[ABSTRACT] {abstract_text}"
        input_text = f"{instruction}\n{abstract}\n</s>"  # 和训练时一致，添加结束符
        return input_text

    def evaluate(self):
        """
        评估模型：生成标题并与真实标题进行对比
        """
        generated_titles = []
        true_titles = []

        # 在测试集上进行推理
        for data_point in self.data["test"]:
            abstract_text = data_point["abstract"]
            true_title = data_point["title"]

            # 准备输入
            input_text = self.prepare_input_for_testing(abstract_text)

            # 编码输入
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.model.device)

            # 生成标题
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=512,  # 设置生成的最大长度
                    num_beams=5,  # 使用束搜索
                    no_repeat_ngram_size=2,  # 防止重复的n-gram
                    eos_token_id=self.tokenizer.eos_token_id,  # 使用 eos_token_id
                    early_stopping=True
                )

            # 解码生成的标题
            generated_title = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # 保存真实标题和生成的标题
            generated_titles.append(generated_title)
            print(generated_title)
            true_titles.append(true_title)

        # 计算评估指标（例如 BLEU, ROUGE, etc.）
        self.calculate_metrics(generated_titles, true_titles)

    def calculate_metrics(self, generated_titles, true_titles):
        """
        计算生成标题与真实标题的评估指标
        """
        from sklearn.metrics import accuracy_score
        # 你可以使用 BLEU、ROUGE 等评估指标，下面使用准确度作为示例
        accuracy = accuracy_score(true_titles, generated_titles)
        print(f"Accuracy: {accuracy:.4f}")

        # 如果需要更高级的评估，例如 BLEU 或 ROUGE，可以使用 `nltk` 或 `rouge_score` 库
        # 例如：计算 BLEU 或 ROUGE 分数
        from nltk.translate.bleu_score import corpus_bleu
        bleu_score = corpus_bleu([[t] for t in true_titles], generated_titles)
        print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
