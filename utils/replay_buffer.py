# utils/replay_buffer.py
import torch
from models.tokenizer import Tokenizer
import random
class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []  # [(score, sample_dict)]
        self.seen = set()  # 存储已经加入的 input_ids 哈希值用于去重
        self.tokenizer = Tokenizer.load_tokenizer()  # 加载 tokenizer

    def _hash(self, sample):
        input_ids = sample["input_ids"]
        # 去掉左侧的 padding（也可以根据具体 tokenizer 设置左/右）
        unpadded = input_ids[input_ids != self.tokenizer.pad_token_id]
        return tuple(unpadded.tolist())

    def add(self, sample, score):
        sample_hash = self._hash(sample)
        if sample_hash in self.seen:
            # print("sample already exists in buffer, skipping addition.")
            return  # 已存在，跳过添加
        decoded_input = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        if len(self.buffer) < self.max_size:
            self.buffer.append((score, sample))
            self.seen.add(sample_hash)
            # print(f"✅ added sample: {decoded_input} with score: {score}")
        else:
            self.buffer.sort(key=lambda x: x[0])  # 升序排列
            if score > self.buffer[0][0]:
                # 移除旧的 sample 的 hash
                old_sample = self.buffer[0][1]
                old_hash = self._hash(old_sample)
                self.seen.discard(old_hash)

                # 替换新样本
                self.buffer[0] = (score, sample)
                self.seen.add(sample_hash)
                # print(f"🔄 replaced with sample: {decoded_input} with score: {score}")

    def get_all(self):
        return [s for _, s in self.buffer]

    def save(self, path):
        torch.save(self.buffer, path)

    def load(self, path, ratio):
        assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"
        all_data = torch.load(path)
        # 按分数降序排序
        sorted_data = sorted(all_data, key=lambda x: x[0], reverse=True)
        # 截取前一定比例的数据
        cutoff = int(len(sorted_data) * ratio)
        self.buffer = sorted_data[:cutoff]
