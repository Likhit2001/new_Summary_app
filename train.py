import os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

def preprocess(example):
    return {
        "input_text": "summarize: " + example["article"],
        "target_text": example["highlights"]
    }

def tokenize(example, tokenizer):
    input_enc = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    target_enc = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

def main():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    print("\n[INFO] Loading dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1]") 
    dataset = dataset.map(preprocess)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=dataset.column_names)

    print("\n[INFO] Starting training...")
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    print("\n[INFO] Training complete.")

    print("[INFO] Merging LoRA weights...")
    model = model.merge_and_unload()

    print("[INFO] Saving model and tokenizer to /opt/ml/model")
    model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    added_token_file = os.path.join("/opt/ml/model", "added_tokens.json")
    if os.path.exists(added_token_file):
        print("[INFO] Removing invalid added_tokens.json...")
        os.remove(added_token_file)

if __name__ == "__main__":
    main()