import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from torch.utils.data import DataLoader
from models.model import TinyModelLoader, LargeModelLoader
from utils.config_loader import load_config
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset
from models.weight_network import WeightNetwork
from models.collaborative_inference import CollaborativeInference
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

tokenizer = Tokenizer.load_tokenizer()

def collate_fn(batch):
    """自定义数据整理函数"""
    texts = [item["text"] for item in batch]
    titles = [item["title"] for item in batch]
    text_encodings = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        padding_side="left"
    )
    title_encodings = tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt"
    )
    # 获取 <eos> token 的 ID
    eos_token_id = tokenizer.eos_token_id
    
    # 确保每个标题序列包含 <eos> token
    labels = title_encodings["input_ids"]
    
    for i in range(labels.size(0)):  # 遍历每个样本
        if labels[i, -1] != eos_token_id:  # 如果最后一个 token 不是 <eos> token
            labels[i, -1] = eos_token_id  # 将最后一个 token 替换为 <eos> token
    
    # 返回字典，包括输入文本的 input_ids、attention_mask 和标签 labels
    return {
        "input_ids": text_encodings["input_ids"].to(device),
        "attention_mask": text_encodings["attention_mask"].to(device),
        "labels": labels.to(device)  # 修改后的标签
    }

def calculate_bleu(predictions, references):
    """计算 BLEU 评分"""
    smoothing_function = SmoothingFunction().method4
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        ref = [ref.split()]
        pred = pred.split()
        bleu_score = sentence_bleu(ref, pred, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

    return sum(bleu_scores) / len(bleu_scores)

def calculate_rouge(predictions, references):
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

path = "results/models/combModel/checkpoint_epoch4.pt"
checkpoint = torch.load(path)

config = load_config()

tiny_model = TinyModelLoader.load_finetuned_model()
large_model = LargeModelLoader.load_model()
large_model.resize_token_embeddings(len(tokenizer))



os.environ["CUDA_VISIBLE_DEVICES"]=config["base"]["device_id"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
data = PaperDataset.format_data_combModel(config["base"]["dataset_path"])

weight_network = WeightNetwork()

collaborative_inference = CollaborativeInference(large_model, tiny_model, weight_network, tokenizer, device)

weight_network.load_state_dict(checkpoint["model_state"])

test_loader = DataLoader(
    data["test"],
    batch_size=config["combModel_training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn  # 修改为测试数据整理函数
)

weight_network.eval()  # 设置模型为评估模式
total_loss = 0.0
num_batches = 0
all_predictions = []
all_references = []

with torch.no_grad():  # 关闭梯度计算
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # 前向传播
        outputs = collaborative_inference.forward(input_ids, attention_mask)
        weighted_logits = outputs["combined_logits"]
        # 计算损失
        loss = calculate_loss(weighted_logits, labels)

        total_loss += loss.item()
        num_batches += 1

        answer_tokens = generated_tokens[0][:len(labels[0])]

        # 获取预测和参考文本
        predictions = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)

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
    rouge_scores = calculate_rouge(all_predictions, all_references)
    bleu_scores = calculate_bleu(all_predictions, all_references)

    print(f"ROUGE-1: {rouge_scores['rouge1']}")
    print(f"ROUGE-2: {rouge_scores['rouge2']}")
    print(f"ROUGE-L: {rouge_scores['rougeL']}")
    print(f"BLEU: {bleu_scores}")