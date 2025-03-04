from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils.config_loader import load_config
import os
import torch

class TinyModelLoader:
    @staticmethod
    def load_model():
        config = load_config()
        os.environ["CUDA_VISIBLE_DEVICES"]=config["base"]["device_id"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["base"]["model_id"],
            quantization_config=bnb_config,
        )
        model.to(device)
        return model
