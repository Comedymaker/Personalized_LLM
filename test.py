from utils.config_loader import load_config
from dataloader.dataset import PaperDataset

config = load_config()
with open(config["base"]["dataset_path"], "r", encoding="utf-8") as f:
    data = json.load(f)

print(data["train"][0])