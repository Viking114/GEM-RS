import torch
from torch.nn import Embedding

class DNN(torch.nn.Module):

    def __init__(self, config, uvec=None, ivec=None):
        print('initializing DNN')
        super(DNN, self).__init__()
        self.config = config
        self.user_num = config['u_num']
        self.item_num = config['i_num']
        self.u_embed_dim = config['embed_dim']
        self.i_embed_dim = config['embed_dim']

        if uvec is not None and ivec is not None:
            print('pre-learnt user/item feature detected')
            print('u:', uvec.shape)
            print('i:', ivec.shape)
            self.u_embed_dim = uvec.shape[1]
            self.i_embed_dim = ivec.shape[1]
            self.u_embed = Embedding(num_embeddings=self.user_num, embedding_dim=self.u_embed_dim)
            self.i_embed = Embedding(num_embeddings=self.item_num, embedding_dim=self.i_embed_dim)
            self.u_embed.weight = torch.nn.Parameter(torch.Tensor(uvec))
            self.i_embed.weight = torch.nn.Parameter(torch.Tensor(ivec))

        else:
            print('no pretrained!')
            self.u_embed = Embedding(num_embeddings=self.user_num, embedding_dim=self.u_embed_dim)
            self.i_embed = Embedding(num_embeddings=self.item_num, embedding_dim=self.i_embed_dim)

        self.NN = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.NN.append(torch.nn.Linear(in_size, out_size))

        self.score_layer = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):

        user_embedding = self.u_embed(user_indices)
        item_embedding = self.i_embed(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.NN))):
            vector = self.NN[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.score_layer(vector)
        rating = self.logistic(logits)
        return rating

class Engine(object):

    def __init__(self, config, uPretrain=None, iPretrain=None):
        self.config = config
        self.loss_fn = torch.nn.BCELoss()
        self.model = DNN(config, uvec=uPretrain, ivec=iPretrain)
        self.opt = self.initialize_opt(self.model, config)
        print(self.model)

    def initialize_opt(self,model,params):

        optimizer = torch.optim.Adam(model.parameters(), lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])
        return optimizer

    def train_single_batch(self, users, items, ratings):
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.loss_fn(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):

        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        print('total loss:',total_loss)


