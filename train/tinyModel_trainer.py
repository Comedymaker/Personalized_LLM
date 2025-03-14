from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from utils.config_loader import load_config
from models.model import TinyModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel


class FineTuner:
    def __init__(self): 
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()
        self.model = TinyModelLoader.load_model()
        self.data = PaperDataset.format_data(self.config["base"]["dataset_path"])
        print(self.data["train"][0])
        # self.tokenized_data = self.data.map(PaperDataset.preprocess_function, batched = True)
        # self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        
    def _get_training_args(self):
        return TrainingArguments(
            output_dir=self._get_output_dir(),
            per_device_train_batch_size=self.config["tinyModel_training"]["batch_size"],
            gradient_accumulation_steps=self.config["tinyModel_training"]["gradient_accumulation_steps"],
            learning_rate=self.config["tinyModel_training"]["learning_rate"],
            num_train_epochs=self.config["tinyModel_training"]["num_epochs"],
            max_steps=self.config["tinyModel_training"]["max_steps"],
            fp16=self.config["tinyModel_training"]["fp16"],
            save_total_limit=self.config["tinyModel_training"]["save_total_limit"],
            save_strategy="epoch",
            logging_steps=10,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            report_to="wandb"
        )
    
    def _get_output_dir(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config['tinyModel_training']['output_dir']}/{timestamp}_{self.config['base']['tiny_model_id'].split('/')[-1]}"

    def run(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            peft_config=self._get_lora_config(),
            args=self._get_training_args(),
            tokenizer=self.tokenizer,
            # data_collator=self.data_collator,
            max_seq_length=512,
        )
        trainer.train()
        trainer.evaluate()
        self._save_model(trainer)

    def _get_lora_config(self):
        from peft import LoraConfig
        return LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            target_modules=self.config["lora"]["target_modules"],
            bias="lora_only",
            task_type="CAUSAL_LM"
        )

    def _save_model(self, trainer):
        trainer.model.save_pretrained(trainer.args.output_dir)
        self.tokenizer.save_pretrained(trainer.args.output_dir)
        print(f"Model saved to: {trainer.args.output_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['base']['tiny_model_id'],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16"
            ),
            device_map="auto",
            torch_dtype=torch.float16
        )

        peft_model = PeftModel.from_pretrained(
            base_model,
            trainer.args.output_dir,  # 本地适配器路径
            device_map="auto"
        )

        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(trainer.args.output_dir + "_merged")
        self.tokenizer.save_pretrained(trainer.args.output_dir + "_merged")

        print(f"完整模型已保存到：{trainer.args.output_dir}_merged")
