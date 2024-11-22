# AI_Tutorial_Example
记录AI算法的实践过程，包括但不限于NLP、推荐、音频、图像、推荐、LLM和多模态等
（谨以此记录学习过程：主要借助一些开源的数据集进行实战操作）


# 工具
* torch
* deepspeed
* transfomers
* modelscope
* peft
* ms-swift
  

# 一、LLM部分
## 1.1 LLM 微调(sft)
* [information extraction (信息抽取)](llm/sft)
* * 金融事件及主体抽取
* * 命名实体识别
* * 文本摘要
* * 雅意信息抽取
* [领域多轮对话微调(multi_turn_conversations)](llm/sft/multi_turn_conversation)
* * 心理咨询多轮对话微调
* [文本纠错(text correct)](llm/sft/text_correct/)
* * csc文本纠错
  
## 1.2 [LLM预训练(pretrained)](llm/pretrained)
    
    采用法律问答数据中的答案部分进行领域预训练

## 1.3 [PEFT微调](llm/peft_tutorial)
* IA3
* LoRA
* p-tuning
* prefix_tuning
* primpt_tuning


## 1.4 [nlp_task](llm/nlp_task)
* generattion_chat
* summarization
* ner

## 1.5 [分布式训练](llm/distributed_train/)
* accelerate
* huggingface trainer
  

## 1.6 [LLM部署](llm/llm_deploy/)


## 1.7 [LLM结构化输出](llm/llm_structure_output)


# 二、[CV（+ 多模态）](cv/)

## 2.1 [车牌识别微调](cv/车牌识别/)


## 2.2 [发票OCR识别](cv/发票OCR要素提取)

## 2.3 [目标检测及OCR](cv/florence2微调)



# 三、RAG

## 模型及工具
* elasticsearch 8.15
* langchain
* langgraph
* vllm
* qwen
* xinference
* bge


## RAG流程
* 数据索引index
* * 数据集进行分块
* * 通过产生Embedding
* * 导入到ES中
  
* 召回
* * Embedding召回
* * BM25召回

* 答案生成
* * 评估相关性，并通过Qwen生成答案。
* * 通过VLLM获取结构化输出
  
* RAG评估
* * 评估框架ragas
  


# 四、Agent

# # 4.1 [agent实战-langgraph](agent/agent_practice/)

# # 4.2 [crewai_agent](agent/crewai_agent/)


