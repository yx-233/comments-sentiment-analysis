import torch
import torch.nn as nn
from  torch.optim import Adam
from cn_sentiment_dataset import get_dataloader
from model import mymodel
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mymodel().to(device)
optim = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
dataloader2 = get_dataloader(train=True, batch_size=32)
# model_path = 'model/model.t'
# optim_path = 'model/optim.t'


model_path = 'model/models_save/model_tem.t'
optim_path = 'model/models_save/optim_tem.t'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    optim.load_state_dict(torch.load(optim_path))


def train(epoch):
    acc_list = []
    loss_list = []
    for idx, (input, target) in enumerate(dataloader2):
        optim.zero_grad()
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = loss_fn(output, target)
        loss_list.append(loss.cpu().item())
        loss.backward()
        optim.step()
        pred = output.cpu().max(dim=1)[-1]  # 获取每行最大值的位置
        cur_acc = pred.eq(target.cpu()).float().mean()  # 计算准确率
        acc_list.append(cur_acc)
        if idx % 80 == 0:
            print("训练损失：{loss}, 准确率： {acc}".format(loss=loss.item(), acc=str(np.mean(acc_list)*100)[:5]+'%'))
            # print("当前训练损失：", loss.item())
        # if np.mean(acc_list) > 0.98:
            # print(98)

            torch.save(model.state_dict(), model_path)  # 保存模型参数
            torch.save(optim.state_dict(), optim_path)  # 保存优化器参数
    return loss_list,[float(i) for i in acc_list] #返回损失和准确率
if __name__ == "__main__":
    for i in range(4):
        train(i)