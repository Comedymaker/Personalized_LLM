# train_weight_network.py

from transformers import TrainingArguments
from models.model import LargeModelLoader, TinyModelLoader
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from utils.config_loader import load_config
from dataloader.dataset import PaperDataset
import torch
from torch.optim import AdamW

class WeightNetworkTrainer:
    def __init__(self): 
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()  # 加载tokenizer
        
        # 加载大模型和小模型（tiny_model）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.large_model = LargeModelLoader.load_model(self.config["base"]["large_model_id"], self.device)
        self.tiny_model = TinyModelLoader.load_model(self.config["base"]["model_id"], self.device)
        
        # 加载权重网络
        self.weight_network = WeightNetwork(input_size=2 * self.large_model.config.hidden_size).to(self.device)
        
        # 训练数据
        self.data = PaperDataset.format_data(self.config["base"]["dataset_path"])

        # 初始化协同推理实例
        self.collaborative_inference = CollaborativeInference(self.large_model, self.tiny_model, self.weight_network)

    def freeze_models(self):
        """冻结大模型和小模型的参数"""
        for param in self.large_model.parameters():
            param.requires_grad = False
        for param in self.tiny_model.parameters():
            param.requires_grad = False

    def train_weight_network(self, num_epochs=5):
        """训练权重网络"""
        self.freeze_models()  # 冻结大模型和小模型

        # 使用优化器训练权重网络
        optimizer = AdamW(self.weight_network.parameters(), lr=1e-5)

        self.weight_network.train()
        for epoch in range(num_epochs):  # 迭代多个 epoch
            print(f"Epoch {epoch + 1}/{num_epochs} training started.")
            for batch in self.data["train"]:
                # 先对数据进行tokenization处理
                texts = batch["text"]  # 假设每个batch是一个包含文本的字典
                encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)  # 如果有的话

                # 使用协同推理获取加权后的logits
                weighted_logits = self.collaborative_inference(input_ids)

                # 计算损失
                loss = torch.nn.CrossEntropyLoss()(weighted_logits.view(-1, weighted_logits.size(-1)), input_ids.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs} training finished.")

        print("Weight network trained")
        self.save_weight_network()

    def save_weight_network(self):
        """保存训练好的权重网络"""
        save_path = self.config['training']['output_dir'] + "/weight_network"
        torch.save(self.weight_network.state_dict(), save_path)
        print(f"Weight network saved to: {save_path}")

if __name__ == "__main__":
    trainer = WeightNetworkTrainer()
    trainer.train_weight_network()
