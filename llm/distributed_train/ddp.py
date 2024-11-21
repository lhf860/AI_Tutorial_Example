import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import BertTokenizer, BertForSequenceClassification
import torch.distributed as dist
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split

from typing import Any



dist.init_process_group(backend="nccl")

import pandas as pd

csv_path = "/root/llm-tutorial/distribute_train_tutorial/ChnSentiCorp_htl_all.csv"

data = pd.read_csv(csv_path).dropna()
print(data.head())


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna()
    
    def __getitem__(self, index) -> Any:
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)



dataset = MyDataset()


# 划分数据集


trainset, validset = random_split(dataset, lengths=[0.9, 0.1],generator=torch.Generator().manual_seed(42))
print("划分数据后的长度", len(trainset), len(validset))



cache_dir = "/root/autodl-tmp/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)

    return inputs




from torch.utils.data import DataLoader
# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


batch_size = 64


trainloader = DataLoader(trainset, batch_size=batch_size, sampler=DistributedSampler(trainset, seed=0), collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=batch_size, sampler=DistributedSampler(validset, seed=0), collate_fn=collate_func)


from torch.optim import Adam
import os

from torch.nn.parallel import DistributedDataParallel


# model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)

if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))   # 自动获取LOCAL_RANK的值，默认与GPU的序号id一致


model = DistributedDataParallel(model)

optimizer = Adam(model.parameters(), lr=1e-5)



def print_rank_0(info):
    # print("print_rank_0: ", os.environ["RANK"])
    if int(os.environ["RANK"]) == 0:
        print(info, "11111111111")
    if int(os.environ["LOCAL_RANK"]) == 1:
        print(info, '22222222222')


def evaluate():
    model.eval()

    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)

            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    dist.all_reduce(acc_num)
    return acc_num / len(validset)



def train(num_epoch=3, log_step=10):
    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        trainloader.sampler.set_epoch(epoch)
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                
                print_rank_0(f"epoch: {epoch}, global_step:{global_step}, loss: {loss}")
            global_step += 1
        acc = evaluate()
        print_rank_0(f"epoch: {epoch}, acc: {acc}")
    
train()





