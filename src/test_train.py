from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
)

print("TrainingArguments initialized successfully")
