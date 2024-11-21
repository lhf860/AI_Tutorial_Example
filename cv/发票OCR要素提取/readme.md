# 发票OCR要素提取

# 数据集
[CHIP2022-医疗清单发票OCR要素提取任务](https://tianchi.aliyun.com/dataset/131815) 中的“大赛1000训练用数据集.zip”

# 微调模型：qwen-vl

# swift微调命令（单GPU）
    SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0  swift sft --model_type qwen2-vl-7b-instruct --model_id_or_path qwen/Qwen2-VL-7B-Instruct --sft_type lora   --dataset processed_dataset/fapiao_train.jsonl --val_dataset processed_dataset/fapiao_test.jsonl --num_train_epochs 2 --output_dir sft_models/qwen2-7b-vl-sft-fapiao --batch_size 1  --eval_steps 5 --quantization_bit 4 --gradient_accumulation_steps 6

# swift微调命令（多GPU）
    SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 NPROC_PER_NODE=10 swift sft --model_type qwen2-vl-7b-instruct --model_id_or_path qwen/Qwen2-VL-7B-Instruct --sft_type lora --dataset processed_dataset/yiliaofapiao_ocr_train.jsonl --val_dataset processed_dataset/yiliaofapiao_ocr_test.jsonl  --num_train_epochs 2  --eval_steps 1000 --deepspeed default-zero2

# 合并训练后的LoRA

    CUDA_VISIBLE_DEVICES=0 swift export --ckpt_dir /root/autodl-tmp/vlm/dataset/sft_models/qwen2-7b-vl-sft-fapiao/qwen2-vl-7b-instruct/v1-20241008-161414/checkpoint-20  --merge_lora true  

