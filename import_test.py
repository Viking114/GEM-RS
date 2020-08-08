from CooccurCore.glove import Glove
from testfolder.corpus import Corpus
from test_data.getdata import getdata, gettext8
from gensim.models import Word2Vec
import tqdm
import pickle
from torch import nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from torch.autograd import Variable
import numpy as np
from torch import optim
import torch as t
from tf_glove.tf_glove import GloVeModel
from mittens import GloVe

# class GloveDataset(Dataset):
#
#     def __init__(self, w, c, data, xMax):
#
#         self.w, self.c, self.x = torch.LongTensor(list(w)), torch.LongTensor(list(c)), torch.Tensor(list(data))
#         self.xMax = xMax
#         self.logx = np.log(self.x)
#
#     # weight function
#     def f_function(self, value, powerValue = 3/4):
#         if value == 0:
#             return torch.tensor(0.0).type(torch.float64)
#         if value > self.xMax:
#             return torch.tensor(1.0).type(torch.float64)
#         else:
#             return np.power((value/self.xMax),powerValue)
#
#     def __getitem__(self, item):
#         # w c value weight
#         return (self.w[item],self.c[item],self.logx[item],self.f_function(self.x[item]))
#
#     def __len__(self):
#         return len(self.x)
#
# class GloVe(nn.Module):
#     """Global Vectors for word embedding."""
#     def __init__(self, vocab_size, emb_dim=50, sparse=False):
#         super(GloVe, self).__init__()
#         # Word embeddings.
#         self.embedding = nn.Embedding(vocab_size, emb_dim, sparse=sparse)
#         self.bias = nn.Embedding(vocab_size, 1, sparse=sparse)
#
#         # Context embeddings.
#         self.embedding_tilde = nn.Embedding(vocab_size, emb_dim, sparse=sparse)
#         self.bias_tilde = nn.Embedding(vocab_size, 1, sparse=sparse)
#
#         # Xavier initialization.
#         initrange = (2.0 / (vocab_size + emb_dim))**0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.embedding_tilde.weight.data.uniform_(-initrange, initrange)
#         # Zero initialization.
#         self.bias.weight.data.zero_()
#         self.bias_tilde.weight.data.zero_()
#
#     def forward(self, w, c, logx, weights):
#
#         center,context = self.embedding(w), self.embedding_tilde(c)
#         bi, bj = self.bias(w), self.bias_tilde(c)
#         out = torch.mul(center,context)
#         out = torch.sum(out, dim=1)
#         bi = bi.flatten()
#         bj = bj.flatten()
#
#         sqerr = torch.pow(out + bi + bj - logx.type(torch.float32),2)
#         rs = torch.mul(weights.type(torch.float32),sqerr)
#         loss = torch.mean(rs)
#         return loss
#
#     def extractEmbedding(self,idToStr):
#         embd = self.embedding.weight.cpu().data.numpy()
#         rs = {}
#         for i,vec in enumerate(embd):
#             rs[idToStr[i]] = vec
#
#         return rs
#
data = getdata(lang='en')
# corpus = Corpus()
# corpus.fit(data, window=3,)

# print('fitting word2vec')
# w2v = Word2Vec(sentences=data, iter=20)

# w, c, x = corpus.to_list()
# data_set = GloveDataset(w, c, x, xMax=100)

# w2v = Word2Vec(sentences=data, size=32, iter=30, min_count=0)


#
# def getCoocurDict(ds):
#     rs = {}
#     for i,j,k in zip(ds['w'],ds['c'],ds['x']):
#         if not int(i) in rs:
#             rs[int(i)] = {}
#         rs[int(i)][int(j)] = int(k)
#     return rs
#
#
# ds = pickle.load(open(r'E:\SocialRSonYelp\WeightedSkipGram\glove_C++\userpref.pkl','br'))
# rs = getCoocurDict(ds)
#
# glove = Glove(corpus.to_dict(), d=32,)
# glove.set_dictionary(corpus.dictionary)
# #
# for epoch in tqdm.tqdm(range(30)):
#     print('start training epoch {}'.format(epoch))
#     err = glove.train(workers=16, batch_size=256, step_size=0.05)
#     # err = glove.train(workers=16, batch_size=256, step_size=0.1)
#     print("epoch %d, error %.3f" % (epoch, err), flush=True)


# model = Glove(rs,d=50,alpha=0.75,x_max=100)
#
# for epoch in range(30):
#      print('hi3')
#      err = model.train(step_size=0.01, workers=10, batch_size=64)

# para = {
#     'lr':0.05
# }
#
# dl = DataLoader(data_set, batch_size=128,shuffle=True)
# glove = GloVe(vocab_size=len(corpus.dictionary),emb_dim=32)
# glove = glove.cuda()
# optimizer = optim.Adam(glove.parameters(),lr=para['lr'])
#
# for i in range(10):
#
#     print('epoch {}'.format(i))
#     total_loss = 0
#     for id, batch in enumerate(dl):
#
#         print('epoch:', i,' batch:', id,'total:', len(dl))
#
#         # if id+1%5 == 0:
#         #     optimizer = optim.Adam(glove.parameters(),lr=para['lr']*0.1)
#         optimizer.zero_grad()
#
#         loss = glove(batch[0].type(t.LongTensor).cuda(),
#                      batch[1].type(t.LongTensor).cuda(),
#                      batch[2].type(t.DoubleTensor).cuda(),
#                      batch[3].type(t.DoubleTensor).cuda(),)
#         total_loss += loss
#         loss.backward()
#         optimizer.step()
#
#     print('total loss is {}'.format(total_loss))

# embed = []
# for w in corpus.dictionary:
#     vec = w2v[w]
#     embed.append(vec)
#
# embed = np.stack(embed)

# tf glove here

model = GloVeModel(embedding_size=32, context_size=10)
model.fit_to_corpus(data)
print('start training')
model.train(num_epochs=200)
# #
# embeds = []
# new_dict = {}
# for idx, w in enumerate(corpus.dictionary):
#     embeds.append(model.embedding_for(w))
#     new_dict[w] = idx
# embeds = np.stack(embeds)
# model.generate_tsne()
# glove_model = GloVe(n=32, max_iter=100)  # 25 is the embedding dimension
# print('start training')
# embeddings = glove_model.fit(corpus.matrix)