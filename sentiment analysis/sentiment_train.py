from trainer import train
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# writer = SummaryWriter('log')
acc_list = []
loss_list = []
for i in tqdm(range(30)):
    loss,acc = train(i)
    acc_list.extend(acc)
    loss_list.extend(loss)


plt.rc('font',family='Times New Roman')
plt.figure(figsize=(8,5))
plt.plot(range(len(acc_list)),acc_list,label = 'training_accuracy')
plt.plot(range(len(loss_list)),loss_list,color='red',linewidth=1,label = 'training_loss')
plt.xlabel('literation times')
plt.ylabel('rate')
plt.title('Result Analysis')
plt.ylim(0,1.2)
plt.legend() # 显示图例
plt.show()
# for idx,loss in enumerate(loss_list):
#     writer.add_scalar('training_loss',loss,idx)
# for idx,acc in enumerate(acc_list):
#     writer.add_scalar('training_accuracy',acc,idx)
# writer.close()






