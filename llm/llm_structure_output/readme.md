# 定制LLM的结构输出

# LLM结构化输出的用途
* LLM答案生成（摒弃无关生成）
* RAG答案生成、置信度、相关性等
* 信息抽取：实体识别、关系抽取
* 多项选择等
* 文本分类
* ...
  

# 结构化输出工具
* [vllm](https://github.com/vllm-project/vllm)
* [outlines](https://github.com/dottxt-ai/outlines)
* [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)
* [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo)
* [instructor](https://github.com/instructor-ai/instructor)

# 1. [vllm结构化输出](vllm_structured_output.ipynb)

# 部署qwen2-7b-instruct(vLLM部署)

    python -m vllm.entrypoints.openai.api_server --served-model-name Qwen/Qwen2-7B-Instruct-AWQ  --model Qwen/Qwen2-7B-Instruct-AWQ --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 

# 结构化生成
    参考：vllm_structured_output.ipynb


# 2.[outlines结构化输出](outlines_structured_output.ipynb)

    outlines支持的结构化输出格式相对较多，也是vLLM支持的后端解码框架之一。

# 3.[lm-format-enforcer结构化输出](lm-format-enforcer_structured_output.ipynb)


# 4. [logits-processor-zoo结构化输出（输出限制）](logits-processor-zoo_structured_output.ipynb)
     此框架主要做的是生成内容的限制，作用于decoding的解码策略、限制输出长度、权重提升等

# 5. [instructor结构化输出](instructor_structured_output.ipynb)