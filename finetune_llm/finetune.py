import os
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import setup_chat_format, SFTTrainer

class EnglishFineTuner:
    def __init__(self, base_model_id, output_dir, train_csv, test_csv):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.system_message = """Bạn là một trợ lý thông minh, hãy trở lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan.
                            Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
                NOTE:  - Hãy chỉ trả lời nếu câu trả lời nằm trong tài liệu được truy xuất ra.
                       - Nếu không tìm thấy câu trả lời trong tài liệu truy xuất ra thì hãy trả về : "no" .
                Context: {context}"""
        self.tokenizer = None
        self.model = None

    def build_chat(self, row):
        return {
            "messages": [
                {"role": "system", "content": self.system_message.format(context=row["context"])},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }

    def load_data(self):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        train_ds = Dataset.from_pandas(train_df).map(self.build_chat)
        test_ds = Dataset.from_pandas(test_df).map(self.build_chat)

        return train_ds, test_ds

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quant_config,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def setup_peft(self):
        return LoraConfig(
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        )

    def setup_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            save_strategy="epoch",
            logging_steps=10,
            bf16=True,
            tf32=True,
            report_to="wandb"
        )

    def train(self):
        train_ds, test_ds = self.load_data()
        self.setup_tokenizer()
        self.setup_model()
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.setup_training_args(),
            train_dataset=train_ds,
            eval_dataset=test_ds,
            peft_config=self.setup_peft(),
            max_seq_length=2048,
            packing=True,
            dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

# Run example
if __name__ == "__main__":
    trainer = EnglishFineTuner(
        base_model_id="1TuanPham/T-VisStar-7B-v0.1",
        output_dir="output",
        train_csv="data/train.csv",
        test_csv="data/test.csv"
    )
    trainer.train()