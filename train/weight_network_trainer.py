# train_weight_network.py
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from torch.optim import AdamW
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



class WeightNetworkTrainer:
    def __init__(self):
        # ...保持原有初始化代码...
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()  # 加载tokenizer
        
        os.environ["CUDA_VISIBLE_DEVICES"]=self.config["base"]["device_id"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载大模型和小模型（tiny_model）
        self.large_model = LargeModelLoader.load_model()
        
        self.large_model.resize_token_embeddings(len(self.tokenizer))

        # self.large_model = TinyModelLoader.load_finetuned_model()
        self.tiny_model = TinyModelLoader.load_finetuned_model()
        
        # 加载权重网络
        self.weight_network = WeightNetwork()
        
        # 训练数据
        self.data = PaperDataset.format_data(self.config["base"]["dataset_path"])

        # 初始化协同推理实例
        self.collaborative_inference = CollaborativeInference(self.large_model, self.tiny_model, self.weight_network, self.tokenizer, self.device)

        # 增加数据加载器
        self.train_loader = DataLoader(
            self.data["train"],
            batch_size=self.config["combModel_training"]["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn  # 新增数据整理函数
        )

        # 增加数据加载器
        self.test_loader = DataLoader(
            self.data["test"],
            batch_size=self.config["combModel_training"]["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn  # 新增数据整理函数
        )
        
        # 优化学习率设置
        self.optimizer = AdamW(
            self.weight_network.parameters(),
            lr=1e-5,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader)*self.config["combModel_training"]["epochs"]
        )


    def collate_fn(self, batch):
        """自定义数据整理函数"""
        texts = [item["text"] for item in batch]
        titles = [item["title"] for item in batch]
        text_encodings = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        title_encodings = self.tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=self.config["base"]["max_length"],
            return_tensors="pt"
        )
        # 获取 <eos> token 的 ID
        eos_token_id = self.tokenizer.eos_token_id
        
        # 确保每个标题序列包含 <eos> token
        labels = title_encodings["input_ids"]
        
        for i in range(labels.size(0)):  # 遍历每个样本
            if labels[i, -1] != eos_token_id:  # 如果最后一个 token 不是 <eos> token
                labels[i, -1] = eos_token_id  # 将最后一个 token 替换为 <eos> token
        
        # 返回字典，包括输入文本的 input_ids、attention_mask 和标签 labels
        return {
            "input_ids": text_encodings["input_ids"].to(self.device),
            "attention_mask": text_encodings["attention_mask"].to(self.device),
            "labels": labels.to(self.device)  # 修改后的标签
        }

    def calculate_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

    def evaluate(self):
        """评估模型在测试集上的性能"""
        self.weight_network.eval()  # 设置模型为评估模式
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():  # 关闭梯度计算
            for batch in self.test_loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # 前向传播
                outputs = self.collaborative_inference.forward(input_ids)
                weighted_logits = outputs["combined_logits"]
                # 计算损失
                loss = self.calculate_loss(weighted_logits, labels)

                total_loss += loss.item()
                num_batches += 1

                # 获取预测和参考文本
                predictions = self.tokenizer.batch_decode(torch.argmax(weighted_logits, dim=-1), skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # 打印当前批次的预测和标签（可选：仅打印前 2 个样本）
                print("\n--- 当前批次示例 ---")
                for pred, ref in zip(predictions[:2], references[:2]):
                    print(f"预测文本: {pred}")
                    print(f"标签文本: {ref}")
                    print("-------------------")
                
                all_predictions.extend(predictions)
                all_references.extend(references)

        avg_loss = total_loss / num_batches
        print(f"Test Loss: {avg_loss}")

        # 计算 ROUGE 和 BLEU 评分
        rouge_scores = self.calculate_rouge(all_predictions, all_references)
        bleu_scores = self.calculate_bleu(all_predictions, all_references)

        print(f"ROUGE-1: {rouge_scores['rouge1']}")
        print(f"ROUGE-2: {rouge_scores['rouge2']}")
        print(f"ROUGE-L: {rouge_scores['rougeL']}")
        print(f"BLEU: {bleu_scores}")

        return avg_loss, rouge_scores, bleu_scores

    def calculate_bleu(self, predictions, references):
        """计算 BLEU 评分"""
        smoothing_function = SmoothingFunction().method4
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            ref = [ref.split()]
            pred = pred.split()
            bleu_score = sentence_bleu(ref, pred, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)

        return sum(bleu_scores) / len(bleu_scores)

    def calculate_rouge(self, predictions, references):
        """计算 ROUGE 评分"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }

        for pred, ref in zip(predictions, references):
            score = scorer.score(pred, ref)
            rouge_scores['rouge1'] += score['rouge1'].fmeasure
            rouge_scores['rouge2'] += score['rouge2'].fmeasure
            rouge_scores['rougeL'] += score['rougeL'].fmeasure

        num_samples = len(predictions)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

        return rouge_scores
    
    def freeze_models(self):
        for param in self.tiny_model.parameters():
            param.requires_grad = False

        for param in self.large_model.parameters():
            param.requires_grad = False

    def train_step(self, batch):
        """单步训练流程优化"""
        # 获取输入和标签
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        self.optimizer.zero_grad()

        # 前向传播（逐token生成）
        outputs = self.collaborative_inference.train_forward(input_ids, labels)
        
        # 提取预测logits
        # 注意：需要根据您的实际实现获取中间logits
        # 假设我们能够获取每个位置的加权logits
        weighted_logits = outputs["combined_logits"]  # 需要修改CollaborativeInference来返回这个值

        decoded_labels = [self.tokenizer.decode(label_ids, skip_special_tokens=True) for label_ids in labels]

        # 计算损失
        loss = self.calculate_loss(weighted_logits, labels)
        
        # 反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        

        return loss.item()

    def train_weight_network(self, num_epochs=5):
        """改进的训练循环"""
        self.weight_network.train()
        self.freeze_models()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                total_loss += loss
                
                # 进度打印
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {avg_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        """保存中间结果"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.weight_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        path = f"{self.config['combModel_training']['output_dir']}/checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
