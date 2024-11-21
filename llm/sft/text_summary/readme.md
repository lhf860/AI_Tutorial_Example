# 文本摘要微调



swift sft --sft_type lora --target_modules ALL --dataset sft_text_summary.jsonl --template_type qwen2_5 --model_id_or_path  Qwen/Qwen2.5-3B-Instruct  --output_dir output_ckpt/llm_qwen2.5-3b-sft-text-summary --num_train_epochs 3 --learning_rate 5e-5  --lr_scheduler_type constant 
