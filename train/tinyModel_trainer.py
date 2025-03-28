from typing import Any, Dict, List, Tuple, Union
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from utils.config_loader import load_config
from models.model import TinyModelLoader, LargeModelLoader
from models.tokenizer import Tokenizer
from dataloader.dataset import PaperDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
import torch.nn as nn
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel


BEGIN_KEY = f"<|system|>"
RESPONSE_KEY = f"<|assistant|>"

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取模型输出和标签
        tokenizer = Tokenizer.load_tokenizer()
        
        input_ids = inputs.get("input_ids")  # 获取输入文本的token IDs
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)  # 解码为文本
        print(f"输入文本：{input_text}")  # 打印输入文本

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # 形状 (batch_size, seq_length, vocab_size)

        # 获取第一个样本的预测和真实标签
        first_sample_logits = logits[0]  # 形状：(seq_len, vocab_size)
        first_sample_labels = labels[0]  # 形状：(seq_len,)

        # 将logits转换为预测的token IDs（取最大概率的索引）
        predicted_ids = torch.argmax(first_sample_logits, dim=-1)  # 形状：(seq_len,)

        # 转换为Python列表
        predicted_ids_list = predicted_ids.cpu().numpy().tolist()
        label_ids_list = first_sample_labels.cpu().numpy().tolist()

        # 解码预测和真实标签
        predicted_text = tokenizer.decode(predicted_ids_list, skip_special_tokens=True)
        label_text = tokenizer.decode(
            [id if id != -100 else tokenizer.pad_token_id for id in label_ids_list],
            skip_special_tokens=True
        )

        print(f"\n--- 第一个样本的预测与真实文本 ---")
        print(f"预测文本：{predicted_text}")
        print(f"真实文本：{label_text}")

        # 展平张量，适配交叉熵损失的输入要求
        # logits.view(-1, logits.size(-1)): (batch_size*seq_length, vocab_size)
        # labels.view(-1): (batch_size*seq_length,)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),  # vocab_size 在最后一维
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        # The torch_call method overrides the same method in the base class and 
        # takes a list of examples as input.  
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()
        # print(f"labels0 before: {labels[0]}")

        # The code then encodes a special token, RESPONSE_KEY_NL, 
        # representing the end of the prompt followed by a newline. 
        # It searches for this token in the sequence of tokens (labels) 
        # and finds its index.
        begin_token_ids = self.tokenizer.encode(BEGIN_KEY)
        response_token_ids = [29914, 25580, 29962]

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            # 滑动窗口方式匹配整个 response token 数组
            for idx in range(len(batch["labels"][i]) - len(response_token_ids) + 1):
                # 打印当前窗口匹配的部分
                current_window = batch["labels"][i][idx:idx + len(response_token_ids)].tolist()

                if current_window == response_token_ids:
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                token_ids = batch["labels"][i].tolist()
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {token_ids}'
                )

            # 修正结束索引的计算
            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            # 更新标签，使得响应前的所有 token 都是 -100
            labels[i, :response_token_ids_start_idx] = -100  # 设置响应前的部分为 -100

        batch["labels"] = labels

        # print(f"labels0 after: {labels[0]}")

        return batch

class FineTuner:
    def __init__(self): 
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()
        self.model = TinyModelLoader.load_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data = PaperDataset.format_data(self.config["base"]["dataset_path"])
        print(self.data["train"][0])
        # self.tokenized_data = self.data.map(PaperDataset.preprocess_function, batched = True)
        self.data_collator = DataCollatorForCompletionLM(tokenizer=self.tokenizer, mlm=False, return_tensors="pt")
        
    def _get_training_args(self):
        return TrainingArguments(
            output_dir=self._get_output_dir(),
            per_device_train_batch_size=self.config["tinyModel_training"]["batch_size"],
            gradient_accumulation_steps=self.config["tinyModel_training"]["gradient_accumulation_steps"],
            learning_rate=self.config["tinyModel_training"]["learning_rate"],
            num_train_epochs=self.config["tinyModel_training"]["num_epochs"],
            max_steps=self.config["tinyModel_training"]["max_steps"],
            fp16=self.config["tinyModel_training"]["fp16"],
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
        messages = [
            {
                "role": "system",
                "content": "You are an academic assistant who always generate a concise and accurate paper title based on the abstract provided by the user, without any explanations or formatting. The title should: 1) capture the core innovation; 2) include key technical terms; 3) be under 20 words.",
            },
            {"role": "user", "content": "Address correlation is a technique that links the addresses that reference the same data values. Using a detailed source-code level analysis, a recent study [1] revealed that different addresses containing the same data can often be correlated at run-time to eliminate on-chip data cache misses. In this paper, we study the upper-bound performance of an Address Correlation System (ACS), and discuss specific optimizations for a realistic hardware implementation. An ACS can effectively eliminate most of the L1 data cache misses by supplying the data from a correlated address already found in the cache to thereby improve the performance of the processor. For 10 of the SPEC CPU2000 benchmarks, 57 to 99% of all L1 data cache load misses can be eliminated, which produces an increase of 0 to 243% in the overall performance of a superscalar processor. We also show that an ACS with 1-2 correlations for a value can usually provide comparable performance results to that of the upper bound. Furthermore, a considerable number of correlations can be found within the same set in the L1 data cache, which suggests that a low-cost ACS implementation is possible. "},
            {"role": "assistant", "content": "Improving Data Cache Performance via Address Correlation: An Upper Bound Study"}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"prompt: {prompt}")

        input_ids = self.tokenizer(prompt, return_tensors="pt")

        print(f"input_ids: {input_ids['input_ids']}")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            peft_config=self._get_lora_config(),
            args=self._get_training_args(),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            # max_seq_length=512,
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
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['base']['tiny_model_id'],
            load_in_8bit=False,
            device_map="auto",
            torch_dtype=torch.float16
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        peft_model = PeftModel.from_pretrained(
            base_model,
            f"{trainer.args.output_dir}/checkpoint-{self.config['tinyModel_training']['max_steps']}",  # 本地适配器路径
            device_map="auto",
            from_transformers=True
        )

        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(trainer.args.output_dir + "_merged")
        self.tokenizer.save_pretrained(trainer.args.output_dir + "_merged")

        print(f"完整模型已保存到：{trainer.args.output_dir}_merged")
