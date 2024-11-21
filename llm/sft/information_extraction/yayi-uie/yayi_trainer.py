"""
参考：  
1. https://github.com/yuanzhoulvpi2017/zero_nlp/wiki/%E4%BB%8Esft_clm_mlm%E4%B8%89%E7%A7%8D%E8%AE%AD%E7%BB%83%E6%96%B9%E5%BC%8F%E6%9D%A5%E7%9C%8Bdata_collator%E2%80%94%E2%80%94%E3%80%90transformers%E6%BA%90%E7%A0%81%E9%98%85%E8%AF%BB%E3%80%91
2. https://huggingface.co/learn/nlp-course/zh-CN/chapter7/6


利用指令微调的数据（包括：instruction、input、output等字段的数据样本）进行持续预训练，进行的是因果语言模型（CausalLM）， 
需要注意的是：在一般的因果语言模型CLM中，input_ids和label_ids偏移一个位置， 然后组成数据进行预训练
在基于指令微调数据的语言模型sft中，input_ids是instruction和input组成的source, output组成的target

"""


import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, sys
import json
import pandas as pd
import numpy as np


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset, Dataset
from pprint import pprint






train_json_path = "/root/autodl-tmp/ie/yayi_uie_sft_data/train_sft.jsonl"

test_json_path = "/root/autodl-tmp/ie/yayi_uie_sft_data/test_sft.jsonl"

dataset = load_dataset("json", data_files=test_json_path, split="train")



# model_name = "Qwen/Qwen2-7B-Instruct"
# cache_dir = "/root/autodl-tmp/qwen2-7b-instruct"


from typing import Dict, List

# 参考qwen2官方参考的finetuning代码，发现是错误的
def process_fun_error(example: Dict):
    """自己实现instruction和response的区分"""
    system_prompt = "你是一个文本信息抽取领域的专家，现在需要你从用户user的给定文本中任务类型完成对应的信息抽取任务"
    instruction = example["instruction"]
    response = example["label"]
    messages = [{"role": "user", "content": instruction}, {"role": "system", "content": labels}]
    tokenized_message_ids = tokenizer.apply_chat_template(messages, 
                                                      tokenize=True, 
                                                      chat_template=tokenizer.chat_template,
                                                      add_generation_prompt=False, 
                                                      padding="max_length", 
                                                      max_length=8192, 
                                                      truncation=True)  # # list[int]
    # print(tokenizer.decode(tokenized_message_ids))
    input_ids = torch.tensor(tokenized_message_ids, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids==tokenizer.pad_token_id] = LabelSmoother.ignore_index  # -100
    attenion_mask = input_ids.ne(tokenizer.pad_token_id).int()   # boolean转int类型
    return dict(input_ids=input_ids, attenion_mask=attenion_mask, target_ids=target_ids)
    


def process_fun_qwen15(example):
    
    instruction = example["instruction"]
    response = example["label"]
    # messages = [{"role": "user", "content": instruction}, {"role": "system", "content": labels}]
    # tokenized_message_ids = tokenizer.apply_chat_template(messages, 
    #                                                   tokenize=True, 
    #                                                   chat_template=tokenizer.chat_template,
    #                                                   add_generation_prompt=False, 
    #                                                   padding="max_length", 
    #                                                   max_length=8192, 
    #                                                   truncation=True)  # # list[int]
    
    template = f"<|im_start|>system\n你是一个文本信息抽取领域的专家，现在需要你从用户user的给定文本中任务类型完成对应的信息抽取任务<|im_end|>\n<|im_start|>user\n{instruction }<|im_end|>\n<|im_start|>assistant\n"
    tokenized_instruction = tokenizer(template, add_special_tokens=False)
    tokenized_response = tokenizer(response, add_special_tokens=False)
    
    input_ids = tokenized_instruction["input_ids"] + tokenized_response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = tokenized_instruction["attention_mask"] + tokenized_response["attention_mask"] + [1]
    labels = [-100] * len(tokenized_instruction["input_ids"]) + tokenized_response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attenion_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}







from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen1.5-7B-Chat"
cache_dir = "/root/autodl-tmp/Qwen1.5-7B-Chat/"
max_length = 8192
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, padding="right")



def main():
    
    
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, 
                                                 # device_map="auto", 
                                                 torch_dtype=torch.bfloat16)


    tokenized_dataset = dataset.map(process_fun_qwen15, remove_columns=dataset.column_names, batched=False)
    print(tokenized_dataset)

    
    


    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                             inference_mode=False,lora_alpha=32, r=8, lora_dropout=0.1)


    # gradient_checkpoint=True, 必须设置: model.enable_input_require_grads()

    gradient_checkpointing = True
    if gradient_checkpointing:
        model.enable_input_require_grads()

    model = get_peft_model(model=model, peft_config=lora_config)

    args = TrainingArguments(output_dir="/root/autodl-tmp/yayi_output/",
                             per_device_eval_batch_size=1, gradient_accumulation_steps=4, 
                             logging_dir="/root/autodl-tmp/yayi_output/logging",logging_steps=10, learning_rate=3e-5, 
                             save_steps=50, save_total_limit=1,
                             num_train_epochs=3, 
                             gradient_checkpointing=gradient_checkpointing)


    trainer = Trainer(model=model, args=args, 
                      data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=max_length, label_pad_token_id=-100),
                      train_dataset=tokenized_dataset)


    trainer.train()


if __name__ == "__main__":
    main()

