# 文本纠错

# 数据集
* [中文文本纠错数据集-SIGHAN2015](https://modelscope.cn/datasets/heaodong/SIGHAN2015/dataPeview)
* [中文文本纠错数据集-Weaxrcks/csc](https://www.modelscope.cn/datasets/Weaxrcks/csc.git)

# 训练数据
本文参考csc 数据集中的train.jsonl进行训练和评估，其中根据总长度进行了过滤，具体参考代码。

# 微调LLM
CUDA_VISIBLE_DEVICES=0,1,2,3,4 NPROC_PER_NODE=5 swift sft --sft_type lora --target_modules ALL --dataset train_sft.jsonl --template_type qwen2_5 --model_id_or_path  Qwen/Qwen2.5-3B-Instruct  --output_dir output_ckpt/llm_qwen2.5-3b-sft-text_correct --num_train_epochs 3 --learning_rate 1e-4  --lr_scheduler_type cosine --eval_steps 500 --max_length 128 --batch_size 32 --deepspeed default-zero2





