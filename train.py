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
    input_enc = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=1024)
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
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]") 
    dataset = dataset.map(preprocess)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=dataset.column_names)

    print("\n[INFO] Starting training...")
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=4,
        num_train_epochs=5,
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

# import os
# from datasets import load_dataset
# from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
# from peft import get_peft_model, LoraConfig, TaskType
# import evaluate
# import numpy as np
# import torch

# def preprocess(example):
#     return {
#         "input_text": "summarize: " + example["article"],
#         "target_text": example["highlights"]
#     }

# def tokenize(example, tokenizer):
#     input_enc = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
#     target_enc = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=128)
#     input_enc["labels"] = target_enc["input_ids"]
#     return input_enc

# def compute_rouge(model, tokenizer, dataset):
#     rouge = evaluate.load("rouge")
#     inputs = tokenizer(dataset["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
#     labels = dataset["target_text"]

#     with torch.no_grad():
#         outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128)

#     preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     refs = labels

#     scores = rouge.compute(predictions=preds, references=refs)
#     print(f"[ROUGE-L] score: {round(scores['rougeL'] * 100, 2)}")

# def main():

#     model_name = "t5-small"
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)

#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         target_modules=["q", "v"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.SEQ_2_SEQ_LM
#     )
#     model = get_peft_model(model, lora_config)

#     print("\n[INFO] Loading dataset...")
#     dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10]")  # small subset
#     dataset = dataset.map(preprocess)
#     dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=dataset.column_names)

#     # Prepare for ROUGE eval
#     eval_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:5]")
#     eval_dataset = eval_dataset.map(preprocess)

#     print("\n[INFO] ROUGE before training:")
#     compute_rouge(model, tokenizer, eval_dataset)

#     print("\n[INFO] Starting training...")
#     training_args = TrainingArguments(
#         output_dir="./model",  # changed from /opt/ml/model
#         per_device_train_batch_size=4,
#         num_train_epochs=1,
#         save_strategy="epoch",
#         logging_dir="./logs",
#         report_to="none"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset
#     )

#     trainer.train()
#     print("\n[INFO] Training complete.")

#     print("[INFO] Merging LoRA weights...")
#     model = model.merge_and_unload()

#     print("\n[INFO] ROUGE after training:")
#     compute_rouge(model, tokenizer, eval_dataset)

#     print("[INFO] Saving model and tokenizer to ./model")
#     model.save_pretrained("./model")
#     tokenizer.save_pretrained("./model")

#     added_token_file = os.path.join("./model", "added_tokens.json")
#     if os.path.exists(added_token_file):
#         print("[INFO] Removing invalid added_tokens.json...")
#         os.remove(added_token_file)

# if __name__ == "__main__":
#     main()