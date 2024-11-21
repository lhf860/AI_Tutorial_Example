# 领域多轮对话微调


# 1. 多轮对话数据介绍

具体参考：swift的数据格式要求 [数据格式要求](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html)

# 1.1  数据来源和预处理

 数据采用 [心理咨询师数字孪生对话数据集](https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus)


# 1.2 数据预处理

    参考：预处理数据.ipynb


# 2.微调命令（sft）

# 2.1 微调命令(错误版)：

    swift sft --sft_type lora --target_modules ALL --dataset llm_pretrain.jsonl  --model_id_or_path  qwen/Qwen2.5-3B --auto_find_batch_size True --output_dir output_ckpt/llm_qwen2.5-3b-pretrained 


# 注意 
    查看预训练、监督微调时，需要针对不同的模型或者模版格式进行设置
    # 查看swift支持的模型如下：

    from swift.llm.utils import template, TEMPLATE_MAPPING
    print(TEMPLATE_MAPPING.keys())
    


# 对于使用qwen2.5的模型进行多轮对话的微调的命令和使用的模版类型如下：
# # 下面这个命令可以正常运行：
# 2.2 微调命令（完整版）
    fswift sft --sft_type lora --target_modules ALL --dataset llm_pretrain.jsonl --template_type qwen2_5 --model_id_or_path  qwen/Qwen2.5-3B  --output_dir output_ckpt/llm_qwen2.5-3b-pretrained 