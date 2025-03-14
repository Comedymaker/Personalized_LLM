# train_weight_network.py

from transformers import TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from torch.optim import AdamW



class WeightNetworkTrainer:
    def __init__(self):
        # ...保持原有初始化代码...
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()  # 加载tokenizer
        
        os.environ["CUDA_VISIBLE_DEVICES"]=self.config["base"]["device_id"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载大模型和小模型（tiny_model）
        self.large_model = LargeModelLoader.load_model()
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
        
        # 优化学习率设置
        self.optimizer = AdamW(
            self.weight_network.parameters(),
            lr=self.config["combModel_training"]["lr"],
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
            padding="longest",
            truncation=True,
            max_length=512,
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
        """改进的损失计算"""
        print(f"logits.shape: {logits.shape}")
        print(f"labels.shape: {labels.shape}")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
    
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
        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1.0)
        
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
