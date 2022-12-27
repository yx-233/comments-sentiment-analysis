from cn_sentiment_dataset import get_dataloader
import torch
import torch.nn as nn
from model import mymodel
import numpy as np
from  torch.optim import Adam
import os
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test(train = False):#不需要计算梯度
    loss_list = []
    acc_list = []
    model.eval()
    test_dataloader = get_dataloader(train=train,batch_size=128)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            output = output.cpu()
            # target = target.cpu()
            cur_loss = loss_fn(output, target.cpu())
            loss_list.append(cur_loss)
            #计算准确率
            #output:[batch_size,10] target:[batch_size]

            pred = output.max(dim=1)[-1]#获取每行最大值的位置
            cur_acc = pred.eq(target.cpu()).float().mean()#计算准确率
            acc_list.append(cur_acc)
    print("测试损失：{loss}, 准确率 {acc}".format(loss=np.mean(loss_list), acc=str(np.mean(acc_list)*100)[:5]+'%'))
    return loss_list, [float(i) for i in acc_list]
    # print('测试,准确率,损失',np.mean(acc_list),np.mean(loss_list))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_path = 'model/model.t'
    # optim_path = 'model/optim.t'
    model_path = 'model/models_save/model_tem.t'
    optim_path = 'model/models_save/optim_tem.t'

    model = mymodel().to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))
    loss_fn = nn.CrossEntropyLoss()
    test()
    # acc_list=[]
    # for i in range(10):
    #     loss,acc = test()
    #     acc_list.extend(acc)
    # print(np.mean(acc_list))
    # #画图
    # plt.rc('font', family='Times New Roman')
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(acc_list)), acc_list, label='test_accuracy')
    # plt.xlabel('literation times')
    # plt.ylabel('rate')
    # plt.title('Result Analysis')
    # plt.ylim(0, 1)
    # plt.legend()  # 显示图例
    # plt.show()
