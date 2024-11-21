

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from datasets import load_dataset


dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train",)

print(dataset)

dataset = dataset.filter(lambda x: x["review"] is not None)

# 在多GPU上运行时，必须设置seed， 保证在多进程上的运行的数据切分是一致的
datasets = dataset.train_test_split(test_size=0.1, seed=42)


import torch


cache_dir = "/root/autodl-tmp/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)

print("model.config: ", model.config)


def process(examples):
    toeknized_examples = tokenizer(examples["review"], max_length=128, padding="max_length", truncation=True)
    toeknized_examples["labels"] = examples["labels"]

    return toeknized_examples


tokenized_datasets = datasets.map(process, remove_columns=datasets.column_names)


import evaluate

acc_metric = evaluate.load("./metric_accuracy.py")
f1_metric = evaluate.load("./metric_f1.py")

def eval_metric(eval_predict):
    predictions, labels = eval_predict

    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


train_args = TrainingArguments(output_dir="/root/autodl-tmp/bert_ddp", 
                               per_device_train_batch_size=32,
                               per_device_eval_batch_size=128,
                               evaluation_strategy="epoch", 
                               save_strategy="epoch"
                               logging_steps=10,
                               save_total_limit=2,
                               learning_rate=2e-5,
                               weight_decay=0.01,
                               metric_for_best_model="f1",
                               load_best_model_at_end=True)



from transformers import DataCollatorWithPadding

trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_datasets["train"], eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
                  compute_metrics=eval_metric)


trainer.train()

















