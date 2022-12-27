import torch
import torch.nn as nn
from lib import ws
import lib
import numpy as np
import torch.nn.functional as F
from utils import SelfAttention

# 全连接模型
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding = nn.Embedding(len(ws),lib.embedding_dim)
#         self.fc1 = nn.Linear(lib.max_len*lib.embedding_dim,10)
#         self.fc2 = nn.Linear(10,5)
#         self.fc3 = nn.Linear(5, 2)
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #(batch_size,max_len,100)
#         x = x.view(-1,lib.max_len*lib.embedding_dim)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         out = self.fc3(x)
#         return out

#双向LSTM模型1
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,bidirectional=True)
#         self.layer = nn.Sequential(
#                                             nn.Linear(lib.hidden_size * 2, 2),
#                                             nn.Dropout(lib.dropout)
#                                  )
#         self.fc1 = nn.Linear(lib.hidden_size*2,2)
#         # self.fc2 = nn.Linear(10,2)
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #(batch_size,max_len,100)
#         x,(h_n,c_n) = self.lstm(x)#x[batch_size,max_len,hidden_size*2],h,c[layers*2,batch_size,hidden_size]
#         #获取两个方向最后一次的output,进行拼接
#         output_fw = h_n[-2,:,:]#正向最后一次输出
#         output_bw = h_n[-1,:,:]#反向最后一次输出
#         output = torch.cat([output_fw,output_bw],dim=-1)#output[batch_size,hidden_size*2]
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out

# 双向LSTM模型2
class mymodel(nn.Module):

    def __init__(self):
        super(mymodel,self).__init__()
        self.embedding_dim = lib.embedding_dim
        self.embedding = nn.Embedding(len(ws),self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
                            batch_first=True,bidirectional=True,dropout=lib.dropout)
        self.layer = nn.Sequential(nn.Linear(lib.hidden_size*2,10),
                                  nn.ReLU(True),
                                  nn.BatchNorm1d(10),
                                  nn.Dropout(lib.dropout),

                                  nn.Linear(10,10),
                                  nn.ReLU(True),
                                  nn.BatchNorm1d(10),
                                  nn.Dropout(lib.dropout),

                                  nn.Linear(10,2),
                                )


    def forward(self,input):
        #input(batch_size,max_len)
        x = self.embedding(input)
        #(batch_size,max_len,embedding_dim)
        x,(h_n,c_n) = self.lstm(x)#x[batch_size,max_len,hidden_size*2],h,c[layers*2,batch_size,hidden_size]
        #获取两个方向最后一次的output,进行拼接
        output_fw = h_n[-2,:,:]#正向最后一次输出
        output_bw = h_n[-1,:,:]#反向最后一次输出
        output = torch.cat([output_fw,output_bw],dim=-1)#output[batch_size,hidden_size*2]
        out = self.layer(output)
        # output = F.relu(output)
        # out = self.fc2(output)
        return out

# 双向LSTM模型3
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.bi_lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,bidirectional=True)
#         self.lstm = nn.LSTM(input_size=lib.hidden_size*2,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,bidirectional=False)
#         self.layer = nn.Sequential(nn.Linear(lib.hidden_size,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,2),
#                                 )
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #(batch_size,max_len,embedding_dim)
#         x,(h_n,c_n) = self.bi_lstm(x)#x[batch_size,max_len,hidden_size*2],h,c[layers*2,batch_size,hidden_size]
#         #获取两个方向最后一次的output,进行拼接
#         output_fw = h_n[-2,:,:]#正向最后一次输出
#         output_bw = h_n[-1,:,:]#反向最后一次输出
#         output = torch.cat([output_fw,output_bw],dim=-1)#output[batch_size,hidden_size*2]
#         output,(h,c) = self.lstm(output)#output[batch_size,hidden_size]
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out

#self_attention+bi_lstm
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.attention = SelfAttention(num_attention_heads=2,hidden_size=lib.embedding_dim,input_size=lib.embedding_dim
#                                        ,hidden_dropout_prob=0.2)
#         self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,bidirectional=True,dropout=lib.dropout)
#         self.layer = nn.Sequential(nn.Linear(lib.hidden_size*2,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,2),
#                                 )
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #x(batch_size,max_len,embedding_dim)
#         x = self.attention(x)
#         # x(batch_size,max_len,embedding_dim)
#         x,(h_n,c_n) = self.lstm(x)#x[batch_size,max_len,hidden_size*2],h,c[layers*2,batch_size,hidden_size]
#         #获取两个方向最后一次的output,进行拼接
#         output_fw = h_n[-2,:,:]#正向最后一次输出
#         output_bw = h_n[-1,:,:]#反向最后一次输出
#         output = torch.cat([output_fw,output_bw],dim=-1)#output[batch_size,hidden_size*2]
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out

#bi_lstm+self_attention
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,bidirectional=True,dropout=lib.dropout)
#         self.attention = SelfAttention(num_attention_heads=4,hidden_size=lib.hidden_size*2,input_size=lib.hidden_size*2
#                                        ,hidden_dropout_prob=0.2)
#         self.layer = nn.Sequential(nn.Linear(lib.hidden_size*2,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,2),
#                                 )
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #x(batch_size,max_len,embedding_dim)
#         x,(h_n,c_n) = self.lstm(x)#x[batch_size,max_len,hidden_size*2],h,c[layers*2,batch_size,hidden_size]
#         #获取两个方向最后一次的output,进行拼接
#         output_fw = h_n[-2,:,:]#正向最后一次输出
#         output_bw = h_n[-1,:,:]#反向最后一次输出
#         output = torch.cat([output_fw,output_bw],dim=-1)
#         #output[batch_size,hidden_size*2]
#         x = x.contiguous().view(-1, 1, lib.hidden_size * 2)
#         #x[batch_size,1,hidden_size*2]
#         x = self.attention(x)
#         # x[batch_size,1,hidden_size*2]
#         x = x.contiguous().view(-1,lib.hidden_size * 2)
#         # x[batch_size,hidden_size*2]
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out
#单向lstm
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
#                             batch_first=True,dropout=lib.dropout)
#         self.layer = nn.Sequential(nn.Linear(lib.hidden_size,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,2),
#                                 )
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         # x (batch_size,max_len,embedding_dim)
#         x,(h,c) = self.lstm(x)#x[batch_size,max_len,hidden_size]
#         output = x[:,-1,:]#[batch_size,hidden_size]
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out

#单self_attention
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.embedding_dim = lib.embedding_dim
#         self.embedding = nn.Embedding(len(ws),self.embedding_dim)
#         self.attention = SelfAttention(num_attention_heads=2,hidden_size=lib.embedding_dim,input_size=lib.embedding_dim
#                                        ,hidden_dropout_prob=0.2)
#         self.layer = nn.Sequential(nn.Linear(lib.max_len*lib.embedding_dim,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,10),
#                                   nn.ReLU(True),
#                                   nn.BatchNorm1d(10),
#                                   nn.Dropout(lib.dropout),
#
#                                   nn.Linear(10,2),
#                                 )
#
#
#     def forward(self,input):
#         #input(batch_size,max_len)
#         x = self.embedding(input)
#         #x(batch_size,max_len,embedding_dim)
#         x = self.attention(x)
#         # x(batch_size,max_len,embedding_dim)
#         output = x.view(-1,lib.max_len*lib.embedding_dim)
#         # x(batch_size, max_len*embedding_dim)
#         out = self.layer(output)
#         # output = F.relu(output)
#         # out = self.fc2(output)
#         return out

if __name__ == '__main__':
    from torchviz import make_dot
    model = mymodel()
    x = torch.LongTensor(np.random.randint(0,10,[10,lib.max_len]))  # 定义一个网络的输入值
    y = model(x)  # 获取网络的预测值
    MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    MyConvNetVis.format = "png"
    # 指定文件生成的文件夹
    MyConvNetVis.directory = "log"
    # 生成文件
    MyConvNetVis.view()

