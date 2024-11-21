
"""
按照accelerate的官方教程，添加4行代码进行DDP训练

本代码是在ddp_refactor.py上更改而来，主要更改的地方如下：
1、数据集上
 1）按照单机处理数据的方式来处理，不需要指定DistributeSampler

2、模型和优化器
   模型和优化，只需要进行单机版的定义即可，不需要迁移模型到GPU设备上及特定的GPU id上（Local_rank）
3、损失计算
   不需要通过手动在各个GPU设备上进行通信，只需要调用accelerator.gather()函数即可

4、模型训练和评估
   模型训练和评估过程中，不需要迁移batch数据到GPU设备上
    模型在各个设备上的损失，可以通过步骤3）获取
     模型评估，不需要关心评估样本是否能够均分，只需要通过accelerator.gather（pred）来获取各个GPU设备上的预测值，然后进行评估
5、GPU集群日志打印
   GPU机器上的日志打印不需要区分GPU设备，可以同构accelerator.print函数来打印想要的信心，用法与print方法一致。

6、不需要初始化进程组

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
import time

# 修改by accelerate

from accelerate import Accelerator

# accelerator = Accelerator()  # 初始化Accelerate, 默认是fp32
# accelerator = Accelerator(mixed_precision="bf16")  # 初始化Accelerate, 启动混合精度训练
accelerator = Accelerator(gradient_accumulation_steps=4)  # 初始化Accelerate, 启动混合精度训练

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


    # trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func,sampler=DistributedSampler(trainset, seed=42))
    # validloader = DataLoader(validset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(validset, seed=42))

    """accelerate 改动， 不要设置sampler了"""
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
    validloader = DataLoader(validset, batch_size=32, shuffle=True, collate_fn=collate_func)

    return trainloader, validloader




def prepare_model_and_optim():


    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", cache_dir=cache_dir)

    """accelerate 改动: 不需要手动将模型指定到具体的GPU id上"""
    # if torch.cuda.is_available():
    #     model = model.to(int(os.environ["LOCAL_RANK"]))
    # """下面这个代码要放到model移动到GPU后"""
    # model = DDP(model)  
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
            """accelerate 改动： 不需要将数据迁移到GPU设备上"""
            # if torch.cuda.is_available():
            #     batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}

            output = model(**batch)

            # pred = output.argmax(axis=-1)

            pred = torch.argmax(output.logits, dim=-1)
            # 获取真正数量的预测值（sampler没有再进行填充，是真是的数量预测）
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))

            acc_num += (pred.long()==refs.long()).float().sum()
        """accelerate 改动： 不再需要手动收集具体的example数量"""
        # dist.all_reduce(acc_num)

        return acc_num / len(validloader.dataset)



def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, trainloader: DataLoader, validloader, num_epoch=3, log_step=10):

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        """accelerate 改动： 在accelerate中不需要设置这个代码了"""
        # trainloader.sampler.set_epoch(epoch)
        for batch in trainloader:
            """accelerate 添加梯度累计功能 上下文"""
            with accelerator.accumulate(model):
                """accelerate改动：不需要将数据移动到特别指定的GPU id"""
                # if torch.cuda.is_available():
                #     batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                """accelerate 改动：需要使用accelerate进行loss反向传播"""
                # loss.backward()
                accelerator.backward(loss=loss)
                optimizer.step()

                # 原始的global_step是模型运行一次，就+1一次，在分布式GPU训练中，需要同步一次梯度，才能算作模型训练了一个step
                if accelerator.sync_gradients:   # 判断模型是否同步梯度
                    # 如果同步梯度一次，那说明到达了一个梯度累计的周期，global_step 需要加1
                    global_step += 1

                    if global_step % log_step == 0:
                        """accelerate 改动： 不要手动手机loss、不需要手动打印各个GPU上的信息，可以让信息汇总后统一打印"""
                        # dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                        # print_rank_0(f"epoch: {epoch}, global_step: {global_step}, loss: {loss}")
                        loss =accelerator.reduce(loss, reduction="mean")
                        accelerator.print(f"epoch: {epoch},global_step: {global_step},  loss: {loss}")
                # global_step += 1  # 因为有梯度累计
        acc = evaluate(model, validloader)
        # print_rank_0(f"evaluate epoch: {epoch}, acc: {acc}")
        accelerator.print(f"evaluate epoch: {epoch}, global_step: {global_step}, acc: {acc}")
    


def main():
    """accelerate 改动： 不需要设置进程组了"""
    # dist.init_process_group(backend="nccl")
    trainloader, validloader = prepare_dataloader()
    model, optimizer = prepare_model_and_optim()
    """accelerate 改动： 使用accelerate进行封装， 自动封装成DDP所需的内容"""
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    start = time.time()
    train(model, optimizer, trainloader, validloader, num_epoch=3, log_step=10)
    end = time.time()
    accelerator.print("全部训练时间：", end - start)

if __name__ == "__main__":
    main()


















