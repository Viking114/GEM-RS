import data_utils as du
from dataset import SampleGenerator

config = {    'num_epoch': 50,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 0.01,
              'user_num': 0,
              'item_num': 0,
              'embed_dim': 32,
              'num_negative': 4,
              'layers': [100,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              }

df,_ = du.loadKaggleMovieSmall()
print('Range of userId is [{}, {}]'.format(df.userId.min(), df.userId.max()))
print('Range of itemId is [{}, {}]'.format(df.itemId.min(), df.itemId.max()))
config['user_num'] = df.userId.max()+1
config['item_num'] = df.itemId.max()+1
# DataLoader for training
sample_generator = SampleGenerator(ratings=df)
evaluate_data = sample_generator.evaluate_data

