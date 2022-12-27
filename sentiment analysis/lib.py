import pickle
import os

path = 'model/ws.pkl'
# if os.path.exists(path):
#     ws = pickle.load(open(path, "rb"))
ws = pickle.load(open(path,'rb'))
max_len=100
hidden_size = 300
num_layers = 2
bidirectirons = True
dropout = 0.6
embedding_dim = 500
attention_probs_dropout_prob = 0.1

