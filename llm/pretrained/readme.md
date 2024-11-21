# 说明文档
    此文档主要讲述使用如何使用ms-swift来预训练LLM，这里指定的是：领域的预训练。
    本文档主要讲解主要流程及代码实践。


# 使用环境(主要)
    * torch 2.4
    * swift v2.6.0
    * deepspeed v0.15.4



# 使用数据集

    数据来自 https://github.com/FudanDISC/DISC-LawLLM?tab=readme-ov-file开源的法律数据，本文将使用法律文档数据中的答案进行领域预训练


# 数据预处理（预训练）

    参考：预训练Qwen2.5.ipynb文件
    
# 预训练数据格式：
    {"response": "样本1"}
    {"response": "样本2"}


# 运行命令

    swift pt --model_type qwen2_5-7b-instruct --template_type qwen2_5 --sft_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --init_lora_weights true --gradient_checkpointing True --template_type AUTO --dataset pretrained_law.jsonl --dataset_test_ratio 0.01 --max_length 2048  --num_train_epochs 1 --learning_rate 1e-4 --lr_scheduler_type cosine --save_total_limit 2 --logging_dir logging_dir








