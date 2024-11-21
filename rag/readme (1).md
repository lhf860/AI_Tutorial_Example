# 说明文档

# # 部署LLM 


# # 导入数据到elasticsearch
# # # 文本分割

# # # 多Embedding



# # 定义检索器retrieval

# # # Embedding检索 + BM25检索

# ## Anthoric 上下文检索，减少幻觉，增加召回率

# # 定义Agent



# # 考虑通过多智能体协作的方式来进行问答（multi-agent）




# ES账号密码


# 参考文档

    https://blog.csdn.net/victor_manches/article/details/136328525

# #  查看xinference 的变量

   xinference list :查看正在运行的LLM
   xinference terminate --model-uid "my-llama-2"：  终止某个运行的程序员
   


# 启动服务

1、 xinference-local --host 0.0.0.0 --port 9997
2、 部署LLM：
     1(不要使用，无量化). xinference launch --model-engine vllm --model-name qwen2.5-instruct --size-in-billions 7 --model-format pytorch 
     2. xinference launch --model-engine Transformers  --model-name qwen2.5-instruct --size-in-billions  --model-format pytorch  --quantization 8-bit
     xinference launch --model-engine Transformers  --model-name qwen2.5-instruct --size-in-billions  --model-format pytorch  --quantization 8-bit
3、部署 Embedding
   xinference launch --model-name bge-base-zh-v1.5 --model-type embedding
3、 es启动： ./bin/elasticsearch -d

# 这条命令出现OOM， 解决办法：https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/deploy/transformers_models/qwen/Qwen2___5-7B-Instruct-AWQ  --served-model-name qwen2_5-7b-instruct --quantization awq

# # 这条命令能够正常启动
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/deploy/transformers_models/qwen/Qwen2___5-7B-Instruct-AWQ  --served-model-name qwen2_5-7b-instruct --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 

# 利用vllm启动qwen1.5 chat-7b-awq
export VLLM_USE_MODELSCOPE=True

python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat-AWQ  --served-model-name Qwen/Qwen1.5-7B-Chat-AWQ --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 



python -m vllm.entrypoints.openai.api_server --served-model-name Qwen/Qwen2-7B-Instruct-AWQ  --model Qwen/Qwen2-7B-Instruct-AWQ --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 




# RAG评测
python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen2-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ     --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 
