from transformers import AutoTokenizer
from utils.config_loader import load_config

class Tokenizer:
    @staticmethod
    def load_tokenizer():
        config = load_config()
        tokenizer = AutoTokenizer.from_pretrained(config["base"]["model_id"])
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
