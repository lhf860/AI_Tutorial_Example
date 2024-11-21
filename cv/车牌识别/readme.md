# 车牌识别

# 数据集：CCPD 2019 （自己下载）



# 微调命令命令（多GPU）
    SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC_PER_NODE=6 swift sft --model_type qwen2-vl-7b-instruct --model_id_or_path qwen/Qwen2-VL-7B-Instruct --sft_type lora  --quantization_bit 4  --dataset processed_dataset/plate_train.jsonl --val_dataset processed_dataset/plate_test.jsonl --num_train_epochs 2 --output_dir sft_models/qwen2-7b-vl-sft-plate --batch_size 2  --eval_steps 200