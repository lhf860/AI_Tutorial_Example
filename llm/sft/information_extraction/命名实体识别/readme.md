# LLM命名实体识别


# 数据集下载

git clone https://www.modelscope.cn/datasets/iic/address-ner-ccks-2021.git

# 运行命令：




swift sft --sft_type lora --target_modules ALL --dataset ner_sft_data.jsonl --template_type qwen2_5 --model_id_or_path  qwen/Qwen2.5-3B  --output_dir output_ckpt/llm_qwen2.5-3b-sft-ie --num_train_epochs 3 --learning_rate 5e-5


# 模型微调：qwen/Qwen2.5-3B
swift sft --sft_type lora --target_modules ALL --dataset ner_sft_data.jsonl --template_type qwen2_5 --model_id_or_path  qwen/Qwen2.5-3B  --output_dir output_ckpt/llm_qwen2.5-3b-sft-ie --num_train_epochs 3 --learning_rate 5e-5  --lr_scheduler_type constant

# 模型：Qwen/Qwen2.5-3B-Instruct  （效果更好，99%以上）

swift sft --sft_type lora --target_modules ALL --dataset ner_sft_data.jsonl --template_type qwen2_5 --model_id_or_path  Qwen/Qwen2.5-3B-Instruct  --output_dir output_ckpt/llm_qwen2.5-3b-sft-ie --num_train_epochs 3 --learning_rate 5e-5  --lr_scheduler_type constant

# DSPY微调LLM实现NER任务




# 部署qwen
export VLLM_USE_MODELSCOPE=True
模型名称： Qwen/Qwen2-7B-Instruct

vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8  --max-model-len 1024 

