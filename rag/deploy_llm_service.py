# coding:utf-8



from transformers import Qwen2ForCausalLM, AutoTokenizer




# xinference launch --model-engine vllm --model-name qwen2.5-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}

from langchain.embeddings import XinferenceEmbeddings
# from langchain_community.embeddings import XinferenceEmbeddings



embed_client = XinferenceEmbeddings(server_url="http://localhost:9997/v1",model_uid="bge-base-zh-v1.5")


print(embed_client.embed_query("我是谁"))

# from xinference.client import Client


# client = Client("http://localhost:9997/v1")

# model_uid = client.launch_model(model_name="bge-base-zh-v1.5", model_type="embedding")

# print(model_uid)



















