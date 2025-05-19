# train_weight_network.py
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset, NewsDataset, RatingDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from torch.optim import AdamW
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import datetime



class WeightNetworkTrainer:
    def __init__(self):
        # ...保持原有初始化代码...
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()  # 加载tokenizer
        
        os.environ["CUDA_VISIBLE_DEVICES"]=self.config["base"]["device_id"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path = f"{self.config['combModel_training']['output_dir']}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        os.makedirs(self.path, exist_ok=True)

        self.num_epochs = self.config["combModel_training"]["num_epochs"]

        # 加载大模型和小模型（tiny_model）
        self.large_model = LargeModelLoader.load_finetuned_model()
        
        # self.large_model.resize_token_embeddings(len(self.tokenizer))

        # self.large_model = TinyModelLoader.load_finetuned_model()
        self.tiny_model = TinyModelLoader.load_finetuned_model()
        
        if(self.config["base"]["tiny_model_id"] == "Qwen/Qwen1.5-0.5B-Chat"):
            ctx_dim = 1024
        else:
            ctx_dim = 2048

        # 加载权重网络
        self.weight_network = WeightNetwork(vocab_size=len(self.tokenizer), hidden_dims=[512, 512], ctx_dim=ctx_dim)
        # self.weight_network = WeightNetwork(vocab_size=len(self.tokenizer), hidden_dims=[512, 512])
        # 训练数据
        self.data = PaperDataset.format_data_combModel()
        # self.data = NewsDataset.format_data_combModel()
        # self.data = RatingDataset.format_data_combModel()

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
            collate_fn=self.collate_fn  # 修改为测试数据整理函数
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
            num_training_steps=len(self.train_loader)*self.config["combModel_training"]["num_epochs"]
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
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="left"
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
        # print(f"logits shape: {logits.shape}")
        # print(f"labels shape: {labels.shape}")
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # return F.cross_entropy(
        #     shift_logits.view(-1, shift_logits.size(-1)),
        #     shift_labels.view(-1),
        #     ignore_index=self.tokenizer.pad_token_id
        # )
        flat_logits = logits.view(-1, logits.size(-1))  # (batch_size * output_length, vocab_size)
        flat_labels = labels.view(-1)                   # (batch_size * output_length)
        
        # 计算交叉熵损失，忽略 PAD token
        return F.cross_entropy(
            flat_logits,
            flat_labels,
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
                attention_mask = batch["attention_mask"]

                # 前向传播
                outputs = self.collaborative_inference.forward(input_ids, attention_mask, use_past=False)
                weighted_logits = outputs["combined_logits"]
                # 计算损失
                loss = self.calculate_loss(weighted_logits, labels)

                total_loss += loss.item()
                num_batches += 1

                # 获取预测和参考文本
                predictions = self.tokenizer.batch_decode(outputs["generated_tokens"], skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # 打印当前批次的预测和标签（可选：仅打印前 2 个样本）
                print("\n--- 当前批次示例 ---")
                for pred, ref in zip(predictions[:4], references[:4]):
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
        attention_mask = batch["attention_mask"]
        
        self.optimizer.zero_grad()

        # 前向传播（逐token生成）
        outputs = self.collaborative_inference.forward(input_ids, attention_mask, labels, use_past=True)
        
        # 提取预测logits
        # 注意：需要根据您的实际实现获取中间logits
        # 假设我们能够获取每个位置的加权logits
        weighted_logits = outputs["combined_logits"]  # 需要修改CollaborativeInference来返回这个值

        print("Logits shape:", weighted_logits.shape)

        for pos in [0, 1]:  # 第一个和第二个 token
            logits = weighted_logits[0, pos]  # 取第一个样本的第 pos 个位置
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=10)
            
            print(f"\n🧠 Token Position {pos}:")
            for i, (token, prob) in enumerate(zip(
                self.tokenizer.convert_ids_to_tokens(topk.indices.tolist()),
                topk.values.tolist()
            )):
                print(f"Top {i+1}: Token = {token}, Probability = {prob:.4f}")

        decoded_labels = [self.tokenizer.decode(label_ids, skip_special_tokens=False) for label_ids in labels]
        print(decoded_labels)

        # 计算损失
        loss = self.calculate_loss(weighted_logits, labels)
        
        # 反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def train_weight_network(self):
        """改进的训练循环"""
        self.weight_network.train()
        self.freeze_models()
        num_epochs = self.num_epochs
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
            print(f"Epoch {epoch} 模型已存储到 {self.path}/checkpoint_epoch{epoch}.pt")
        

        replay_path = f"../autodl-tmp/replay_data/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(replay_path, exist_ok=True)
        self.collaborative_inference.replay_buffer.save(f"{replay_path}/replay.pt")
        print(f"重放数据已保存到 ../autodl-tmp/replay_data/{datetime.now().strftime('%Y%m%d%H%M%S')}/replay.pt")

    def save_checkpoint(self, epoch):
        """保存中间结果"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.weight_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        full_path = os.path.join(self.path, f"checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint, full_path)
