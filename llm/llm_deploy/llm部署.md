# LLM部署说明



# 相关工具：
* vllm
* transformers
* sglang
* xinference
* ollama
* tensorrt-llm

# 以qwen2.5-7b举例说明（单卡24G显存）
##  vllm 部署
    vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8  --max-model-len 1024 


