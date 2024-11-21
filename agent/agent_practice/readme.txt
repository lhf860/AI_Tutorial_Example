# vllm 启动LLM
# # 参考： https://qwen.readthedocs.io/zh-cn/stable/deployment/vllm.html
1.  vllm serve Qwen/Qwen2.5-7B-Instruct          【17G显存】
2. python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ \
    --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 
    【8G多】
3. 启动书生浦语
   vllm  serve Shanghai_AI_Laboratory/internlm2_5-7b-chat --trust_remote_code --served-model-name internlm2_5-7b-chat  --gpu-memory-utilization 0.9 --max-model-len 2048 

# 无法启动
vllm serve --model Qwen/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct-AWQ --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 

python -m vllm.entrypoints.openai.api_server --model "/root/autodl-tmp/deploy/transformers_models/qwen/Qwen2___5-7B-Instruct-AWQ"  --served-model-name qwen2_5-7b-instruct --quantization awq --gpu-memory-utilization 0.4 --max-model-len 2048 



# langsmith的API KEY： lsv2_pt_c4b7

# 安装unstructured: pip install "unstructured[docx,pptx, pdf]"





# pip install langchain 0.3.4, langchain_core 0.3.12 【必须langchain 0.3以上】



# redis 安装和启动

安装官网下载redis，并编译和安装，最后启动redis。



