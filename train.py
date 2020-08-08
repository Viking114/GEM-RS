import data_utils as du
from dataset import SampleGenerator
from preprocessing import getSimiGraph
import os
from NN import DNN, Engine
from eval import evaluate
from walker import walk
from feature_extraction import Embeder
import pickle

config = {'num_epoch': 50,
          'batch_size': 256,  # 1024,
          'optimizer': 'adam',
          'adam_lr': 0.01,
          'user_num': 0,
          'item_num': 0,
          'embed_dim': 32,
          'num_negative': 4,
          'layers': [64, 32, 16, 8],
          'l2_regularization': 0.0000001,
          'use_cuda': True,
          'device_id': 0,
          'name': 'test',
          'cache_path': './cache/'
          }

df, _ = du.loadKaggleMovieSmall()
print('Range of userId is [{}, {}]'.format(df.userId.min(), df.userId.max()))
print('Range of itemId is [{}, {}]'.format(df.itemId.min(), df.itemId.max()))
config['user_num'] = df.userId.max() + 1
config['item_num'] = df.itemId.max() + 1
# DataLoader for training
sample_generator = SampleGenerator(ratings=df)
evaluate_data = sample_generator.evaluate_data

u_graph_path = config['cache_path'] + config['name'] + '_user_g.pkl'
i_graph_path = config['cache_path'] + config['name'] + '_item_g.pkl'

# get graph
if not os.path.exists(u_graph_path):
    u_graph = getSimiGraph(df, Type='user', smplRate=1)
    pickle.dump(u_graph, open(u_graph_path, 'bw'))
else:
    u_graph = pickle.load(open(u_graph_path, 'br'))

if not os.path.exists(i_graph_path):
    i_graph = getSimiGraph(df, Type='item', smplRate=1)
    pickle.dump(i_graph, open(i_graph_path, 'bw'))
else:
    i_graph = pickle.load(open(i_graph_path, 'br'))

# get walk
u_walk_path = config['cache_path'] + config['name'] + '_user_w.pkl'
i_walk_path = config['cache_path'] + config['name'] + '_item_w.pkl'

if not os.path.exists(u_walk_path):
    u_walk = walk(u_graph, )
    pickle.dump(u_walk, open(u_walk_path, 'bw'))
else:
    u_walk = pickle.load(open(u_walk_path, 'br'))

if not os.path.exists(i_walk_path):
    i_walk = walk(i_graph)
    pickle.dump(i_walk, open(i_walk_path, 'bw'))
else:
    i_walk = pickle.load(open(i_walk_path, 'br'))

embed = Embeder()

if os.path.exists(config['cache_path'] + config['name'] + '_item_embed.pkl2'):
    i_embed = pickle.load(open(config['cache_path'] + config['name'] + '_item_embed.pkl2', 'br'))
else:
    i_embed = embed.embed(i_walk, 20, embedSize=32, iter=50, Num=config['item_num'])

u_embed = embed.embed(u_walk, 20, 32, 30, config['user_num'])

# pickle.dump(i_embed, open(config['cache_path']+config['name']+'_item_embed.pkl3', 'bw'))

engine = Engine(config, uPretrain=u_embed, iPretrain=i_embed)
#
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = evaluate(model=engine.model, evaluate_data=evaluate_data, epoch_id=epoch,
                               use_cuda=config['use_cuda'])
    print('record written!')
    print('hit_rate', hit_ratio)
    print('ndcg', ndcg)

