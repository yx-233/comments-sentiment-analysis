import numpy as np
from lib import ws
import re
import torch
import torch.nn as nn
from lib import max_len,attention_probs_dropout_prob
import torch.nn.functional as F
import jieba
from word2seq import Word2Sequence
import math
import gensim
"""
文本序列化
"""
def tokenlize(sentence):

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', ',', '-', '\.', '/', ':', ';', '<', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    sentence = sentence.lower() #把大写转化为小写
    sentence = re.sub("<br />"," ",sentence)
    sentence = re.sub("|".join(fileters)," ",sentence)
    result = [i for i in sentence.split(" ") if len(i)>0]

    return result
def cn_words_to_word(sentence):
    fileter = ['\[','\]','\n','\。','\，','\,','\.',"'"]
    sentence = str(list(sentence))
    sentence = list(re.sub("|".join(fileter)," ",sentence).replace(' ',''))
    return sentence

def cn_tokenlize(sentence):
    dic_file = 'dictionary/dict.txt'
    stop_file = 'dictionary/stopwords.txt'
    jieba.load_userdict(dic_file)
    jieba.initialize()
    stopword_list = open(stop_file, encoding='utf-8')
    stop_list = []
    # 过滤停用词词性
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    seg = []
    words = [k for k in jieba.lcut(sentence, cut_all=False)]
    for i in words:
        k = 1
        for j in stop_list:
            #if i == j or len(i) < 2:
            if i == j:
                k = 0
                break
        if k == 1:
            seg.append(i)
    return seg


def collate_fn(batch):
    content,label = list(zip(*batch))
    content = [ws.transform(i,max_len = max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return  content,label

def create_ws(words = False):
    ws = Word2Sequence()
    data_path = 'data/词库.txt'
    with open(data_path, encoding='UTF-8') as f:
        sentence = f.readlines()
    for text in sentence:
        if words:
            text = cn_tokenlize(text)#结巴分词
        else:
            text = cn_words_to_word(text)  # 直接分字
        ws.fit(text)
    ws.build_vocab(min_count=1)
    return ws

def build_word2vec(fname, word2id, save_to_path=None):
    """
    @description: 返回语料文本中词汇集对应的word2vec向量
    @param {*}
    - fname: str, 预训练的word2vec.
    - word2id: dict, 语料文本中包含的词汇集.
    - save_to_path: str, 保存训练语料库中的词组对应的word2vec到本地.
    @return {*}
    语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """

    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1.0, 1.0, [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, "w", encoding="utf-8") as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(" ".join(vec))
                f.write("\n")
    return word_vecs


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


if __name__ == '__main__':
    # ws = ws
    # word2id = ws.dict
    # print(word2id)
    # sentences  = '今天天气真好，去哪里吃饭呢'
    # print(cn_words_to_word(sentences))
    # build_word2vec(,ws.dict,)
    x = torch.LongTensor(np.arange(24).reshape(2,3,4))
    print(x)
    q_par = torch.LongTensor(np.ones([4,4]))
    print(q_par)
    print(torch.matmul(x, q_par))



