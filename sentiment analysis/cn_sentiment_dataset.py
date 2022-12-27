import torch
from torch.utils.data import Dataset,DataLoader
import os
from utils import cn_tokenlize,cn_words_to_word
from lib import max_len,ws
from utils import create_ws
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# max_len = 100
class dataset(Dataset):
    def __init__(self, type, train=True,words=False):
        self.words = words
        self.train = train
        data_path = "data/train" if train else "data/test"
        self.type = type
        self.data_path = os.path.join(data_path,'pos.txt') if self.type == 'pos' else os.path.join(data_path,'neg.txt')

    def __getitem__(self, idx):
        if self.train:
            encoding = 'UTF-8'
        else:
            encoding = 'gbk'
        with open(self.data_path,encoding=encoding) as f:
            self.text = f.readlines()
        if self.words:
            content = cn_tokenlize(self.text[idx])[0:-1]#结巴分词
        else:
            content = cn_words_to_word(self.text[idx])[0:-1]#直接分字
        label = 1 if self.type == 'pos' else 0
        return content, label

    def __len__(self):
        return len(self.text)

def get_dataloader(train = True,batch_size = 4,words = False):
    postive_set = dataset('pos',train=train,words=words)
    postive_set[0]
    # print(len(postive_set))
    negative_set = dataset('neg',train=train,words=words)
    negative_set[0]
    # print(len(negative_set))
    sentiment_set = postive_set + negative_set
    sentiment_set[0]
    # print(len(sentiment_set))
    data_loader = DataLoader(sentiment_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    return data_loader

postive_set = dataset('pos')
postive_set[0]
# print(len(postive_set))
negative_set = dataset('neg')
negative_set[0]
# print(len(negative_set))
sentiment_set = postive_set + negative_set
sentiment_set[0]
# print(len(sentiment_set))
#创建ws


#
def collate_fn(batch):
    content,label = list(zip(*batch))
    content = [ws.transform(i,max_len = max_len) for i in content]
    # content = torch.LongTensor(content).to(device)
    # label = torch.LongTensor(label).to(device)
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return  content,label


dataloader = DataLoader(sentiment_set,batch_size=128,shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
     # fileters = ['"', '#', '$', '%', '&', '\(', '\)', '\*', ',', '-', '\.', '/', ':', ';', '<', '>',"\'"
    #         , '@', '\[','\]', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    # comments=[]
    # for index,(input,target) in enumerate(dataloader):
    #     print(index)
    #     print(input)
    #     # for i in input:
    # #         s = str(ws.inverse_transform(list(i.numpy())))
    # #         s1 = re.sub("<PAD>", "", s)
    # #         s1 = re.sub("|".join(fileters), "", s1)
    # #         s1 = "".join(s1.split(" "))
    # #         comments.append(s1)
    # #         print(s1)
    # #     # print(ws.inverse_transform(list(input[0])))
    #     print(target)
    #     break
    # senti = pd.Series([1,0,1,0],name='sentiment')
    # com = pd.Series(comments,name='comments')
    # df = pd.concat([com,senti],axis=1,ignore_index=False)
    # df.to_excel(r"C:\Users\18079\Desktop\other\pythonProject3\cn_sentiment\汇总.xlsx", sheet_name="sheet1", index= True,encoding="utf-8")
    # data = get_dataloader(train=True,batch_size=100,words=True)
    # print(len(data))
    # datas = dataset('neg',words=True)
    # print(datas[0])
    #创建ws
    path = 'model/ws.pkl'
    if os.path.exists(path):
        pass
    else:
        ws = create_ws()
        pickle.dump(ws, open(path, 'wb'))
    # ws_word = create_ws()
    # ws_words = create_ws(True)
    print(len(ws))
    ws = create_ws()

    print(ws.inverse_dict)
    # print(len(ws_word))
    # print(len(ws_words))
