from lib import ws,max_len
import torch
import pandas as pd
from utils import cn_tokenlize,cn_words_to_word
from model import mymodel
import os
from  torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

#在训练和测试时注意，使用的device为gpu，get_dataloader,collfn也为cpu
#若需要分词处理，创建dataloader,ws,coll_fn时需要设置words为True
#进行预测
#加载模型
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def pre_fn(file_name,file_len=2000,batch_size=500,save = False,words=False):#默认不保存文件，当文件过大时，分批次处理,默认批次为500，
                                                            # file_len为数据数量，要为batch_size的倍数,默认分字处理
    '''
    :param file_name: 文件名，需要是txt
    :param file_len: 需要计算的文件长度，需要是batch_size的整数倍
    :param batch_size: 批处理大小
    :param save: 是否保存为同名的Excel文件
    :param words: 是否分词处理
    :return: 返回一个dataframe
    '''
    model.eval()#设置为eval模式
    comments = []  # 保存评论文本
    sentiment = []  # 保存输出的情感类别
    df = pd.DataFrame()
    with open(file_name, encoding='UTF-8') as f:
        text = f.readlines()
    # 创建批量
    input_t = 'initial'
    batch_size = batch_size
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    if len(text) > 500:  # 对大于500行的数据，每500一个批次处理,默认保留2500行，注意，数据的行数需要是100的整数倍
        # text_tem = text[:batch_size]
        leng = batch_size #每一批次的长度，batch_size
        idx = 0
        while (leng <= file_len):
            text_tem = text[batch_size*idx:batch_size*(idx+1)]
            input_t = 'initial'
            comments = []
            sentiment = []
            for sentence in tqdm(text_tem):#打包批量输入数据[batch_size,max_len]

                if input_t == 'initial':  # 处理第一行评论
                    # print(sentence)
                    sentence = sentence.replace('\n', '')
                    comments.append(sentence)  # 对每一行评论进行遍历，保存到列表中
                    if words:
                        sentence = cn_tokenlize(sentence)  # 分词
                    else:
                        sentence = cn_words_to_word(sentence)#分字,[max_len]
                    input = torch.LongTensor(ws.transform(sentence, max_len)).view(-1,
                                                                                   max_len)  # 将评论转化为模型可以识别的longtensor
                    input_t = input
                else:  # 处理之后的评论
                    sentence = sentence.replace('\n', '')
                    comments.append(sentence)  # 对每一行评论进行遍历，保存到列表中
                    if words:
                        sentence = cn_tokenlize(sentence)  # 分词
                    else:
                        sentence = cn_words_to_word(sentence)#分字,[max_len]
                    input = torch.LongTensor(ws.transform(sentence, max_len)).view(-1, max_len)#tensor(max_len)
                    input_t = torch.cat((input_t, input), dim=0)#tensor(batch_size,max_len)

            idx += 1
            leng = leng + batch_size


            if len(df) == 0:  # 对于第一个批次，创建一个dataframe保存结果
                output = model(input_t)
                output = F.softmax(output, dim=1)  # 将结果转化为概率
                output = output.max(dim=1)[-1]  # 取出概率最大元素的位置
                sentiment = list(output.numpy())
                #汇总结果
                df = pd.DataFrame({'comments':comments,'sentiment':sentiment})
            else:  # 将所有批次结果保存
                df_tem = df
                output = model(input_t)
                output = F.softmax(output, dim=1)  # 将结果转化为概率
                output = output.max(dim=1)[-1]  # 取出概率最大元素的位置
                sentiment = list(output.numpy())
                 # 将结果汇总
                df = pd.DataFrame({'comments': comments, 'sentiment': sentiment})
                df = pd.concat([df_tem, df], axis=0, ignore_index=False)  # 拼接分批次的结果


    else:
        for sentence in tqdm(text):
            if input_t == 'initial':  # 处理第一行评论
                # print(sentence)
                sentence = sentence.replace('\n', '')
                comments.append(sentence)  # 对每一行评论进行遍历，保存到列表中
                if words:
                    sentence = cn_tokenlize(sentence)  # 分词
                else:
                    sentence = cn_words_to_word(sentence)  # 分字,[max_len]
                input = torch.LongTensor(ws.transform(sentence, max_len)).view(-1, max_len)  # 将评论转化为模型可以识别的longtensor
                input_t = input
            else:  # 处理之后的评论
                sentence = sentence.replace('\n', '')
                comments.append(sentence)  # 对每一行评论进行遍历，保存到列表中
                if words:
                    sentence = cn_tokenlize(sentence)  # 分词
                else:
                    sentence = cn_words_to_word(sentence)  # 分字,[max_len]
                input = torch.LongTensor(ws.transform(sentence, max_len)).view(-1, max_len)
                input_t = torch.cat((input_t, input), dim=0)
        output = model(input_t)
        output = output
        # output = F.softmax(output, dim=1)  # 将结果转化为概率
        output = output.max(dim=1)[-1]  # 取出概率最大元素的位置
        sentiment = list(output.numpy())
        # 保存并输出结果
        df = pd.DataFrame({'comments': comments, 'sentiment': sentiment})
    if save:
        df.to_excel(file_name[:-3]+'xlsx', sheet_name="sheet1", index=False, encoding="utf-8")
    return  df
# 首先，它导入了一些必要的库和模块，包括 PyTorch，pandas，自定义的 cn_tokenlize 函数，自定义的 mymodel 类，Adam 优化器，以及 tqdm。
# 然后，它读取保存在模型文件和优化器文件中的模型和优化器的状态，并将它们加载到 model 和 optimizer 变量中。
# 接着，它定义了一个名为 pre_fn 的函数，该函数用于处理给定文件中的文本并预测它们的情感类别。该函数接受两个可选参数：save 和 file_len。如果 save 设置为 True，则会保存预测的结果到文件中，如果 file_len 设置为大于 0 的整数，则会保留最多 file_len 条预测结果。
# 在 pre_fn 函数内部，它首先打开给定的文件并将其中的所有行读入到一个列表中。然后，它会进行以下操作：
# 对于输入文件中的每一行文本，它会使用 cn_tokenlize 函数将其分词，并使用 ws 工具将分词后的文本转换成一个长度为 max_len 的序列。
# 使用 torch.LongTensor 将这个序列转换成一个 PyTorch 张量，并使用 .view() 方法将其转换成一个二维张量。
# 如果这是第一行文本，则将转换后的张量赋值给 input_t 变量。
# 如果不是第一行文本，则使用 torch.cat() 函数将 input_t 和当前转换后的张量拼接在一起，并将拼接后的张量赋值给 input_t。
# 最后，使用 model 对象的 forward() 方法在模型中对 input_t 进行预测，并使用 F.softmax() 函数将模型的输出转换为概率分布。最后，将预测的结果保存到 comments 和 sentiment 列表中。
# 如果输入文件的行数大于 500，则 pre_fn 函数会将文件分成多个批次进行处理，每个批次处理 500 行文本。处理完一个批次后，它会更新文件的起始位置并进行下一个批次的处理，直到处理完所有行为止。
# 最后，如果 save 参数设置为 True，则会将处理的结果保存到一个新的 CSV 文件中。
if __name__ == "__main__":
    model_path = 'model/model.t'
    optim_path = 'model/optim.t'
    model = mymodel()
    optimizer = Adam(model.parameters(), lr=0.001)

    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optim_path))
    # print(pre_fn("data/test/草稿.txt"))
    # pre_fn("data/train/neg.txt",save=True,file_len=1111)
    # df = pre_fn("data/test/草稿.txt",save=True)
    # print(df['sentiment'].value_counts())
    # print(df.loc[df['sentiment']])
    # pre_fn(file_name='data/pred/comments_.txt',file_len=2960,batch_size=296,save = True)
    print(pre_fn("data/test/草稿.txt",file_len=1574,batch_size=787)['sentiment'].value_counts())