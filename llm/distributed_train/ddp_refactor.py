
"""
按照accelerate的官方教程，添加4行代码进行DDP训练

"""


import os

import torch 
import pandas as pd
from torch.optim import Adam
import  torch.distributed  as dist
from torch.utils.data import DistributedSampler, Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification



class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


cache_dir = "/root/autodl-tmp/bert-base-chinese"
# tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)
# model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)


def prepare_dataloader():
    dataset = MyDataset()
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)
    

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs


    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func,sampler=DistributedSampler(trainset, seed=42))
    validloader = DataLoader(validset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(validset, seed=42))
    return trainloader, validloader




def prepare_model_and_optim():


    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)

    
    if torch.cuda.is_available():
        model = model.to(int(os.environ["LOCAL_RANK"]))

    """下面这个代码要放到model移动到GPU后"""
    model = DDP(model)  
    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)




def evaluate(model: torch.nn.Module, validloader):
    model.eval()

    acc_num = 0

    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}

            output = model(**batch)

            # pred = output.argmax(axis=-1)

            prd = torch.argmax(pred, dim=-1)

            acc_num += (pred.long()==batch["labels"].long()).float().sum()
        """注意此代码"""
        dist.all_reduce(acc_num)

        return acc_num / len(validloader)



def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, trainloader: DataLoader, validloader, num_epoch=3, log_step=10):

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        """注意此代码"""
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
                """注意此代码"""
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                print_rank_0(f"epoch: {epoch}, global_step: {global_step}, loss: {loss}")
            global_step += 1
        acc = evaluate(model, validloader)
        print_rank_0(f"evaluate epoch: {epoch}, acc: {acc}")
    


def main():
    dist.init_process_group(backend="nccl")
    trainloader, validloader = prepare_dataloader()
    model, optimizer = prepare_model_and_optim()
    train(model, optimizer, trainloader, validloader, num_epoch=3, log_step=10)


if __name__ == "__main__":
    main()


















