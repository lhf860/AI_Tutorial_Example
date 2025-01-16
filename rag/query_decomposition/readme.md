# LLM用于问题分解--子问题分解



# 模型部署

* 部署Qwen2.5-7B-Instruct
   
    vllm serve  Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct

* 部署Qwen2.5-32B-Instruct-AWQ
  
    llm serve  Qwen/Qwen2.5-32B-Instruct-AWQ --served-model-name Qwen2.5-7B-Instruct 

* 部署书生浦语
  * 部署internlm2_5-7b-chat： 
  
        lmdeploy serve api_server --backend pytorch  --eager-mode Shanghai_AI_Laboratory/internlm2_5-7b-chat  --model-name internlm2_5-7b-chat --model-name Qwen2.5-7B-Instruct 


  * 部署internlm3-8B-instruct
        
        lmdeploy serve api_server --backend pytorch  --eager-mode Shanghai_AI_Laboratory/internlm3-8b-instruct  --model-name internlm2_5-7b-chat --model-name Qwen2.5-7B-Instruct --server-port 8000
* 





