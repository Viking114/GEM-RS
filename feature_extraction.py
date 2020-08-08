from gensim.models import Word2Vec
import torch
import numpy as np
from CooccurCore.glove import Glove
from testfolder.corpus import Corpus
import tqdm
from tf_glove.tf_glove import GloVeModel

class Embeder(object):

    def __init__(self, embed_type=1):
        """
        :param embed_type: 1 SGNS
               embed_type: 2 co-occur C++
               embed_type: 3 co-occur tf
               embed_type 1 is recommended because it runs much faster and easy to reproduce result
        """
        assert embed_type in [1, 2, 3]
        self.embed_type = embed_type

    def embed(self, seq, MaxWindow, embedSize=20, iter=30, Num=0):
        if self.embed_type == 1:
            return self.getEmbd(seq, MaxWindow, embedSize, iter, Num)
        elif self.embed_type == 2:
            return self.getEmbd2(seq, MaxWindow, embedSize, iter, Num)
        else:
            return self.getEmbed3(seq, MaxWindow, embedSize, iter, Num)

    def getEmbd(self, seq, MaxWindow, embedSize=20, iter=30, Num=0):
        print('start to produce Embd with embd size', embedSize)
        model = Word2Vec(sentences=seq, size=embedSize, window=MaxWindow, min_count=0, iter=iter, workers=12)
        rs = []
        for i in range(Num):
            try:
                rs.append(model.wv[str(i)])
            except:
                rs.append(torch.nn.init.xavier_uniform_(torch.Tensor(1, embedSize)).data.numpy()[0])
        return np.array(rs)

    def getEmbed2(self, seq, MaxWindow, embedSize=20, iter=100, Num=0):

        model = GloVeModel(embedding_size=embedSize,context_size=MaxWindow,batch_size=512)
        model.fit_to_corpus(seq)
        model.train(iter)
        return model.embeddings

    def getEmbed3(self,seq, MaxWindow, embedSize=20, iter=200, Num=0):
        print('computing corpus')
        corpus = Corpus()
        corpus.fit(seq,window=MaxWindow)
        model = Glove(corpus.to_dict(), d=embedSize)
        for epoch in tqdm.tqdm(range(iter)):
            print('start training epoch {}'.format(epoch))
            err = model.train(workers=16, batch_size=10000, step_size=0.05)
            # err = glove.train(workers=16, batch_size=256, step_size=0.1)
            print("epoch %d, error %.3f" % (epoch, err), flush=True)

        rs = []
        for i in range(Num):
            try:
                rs.append(model[str(i)])
            except:
                rs.append(torch.nn.init.xavier_uniform_(torch.Tensor(1, embedSize)).data.numpy()[0])

        return np.stack(rs)
