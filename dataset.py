import torch
import random
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
random.seed(0)

"""
From https://github.com/yihong-chen/neural-collaborative-filtering
"""

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        print('weijings data loader')
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.normalize_ratings = self._normalize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        # self.negatives = self._sample_negative(ratings)
        self.rated = self.getRated()
        self.train_ratings, self.test_ratings = self._split_loo(self.normalize_ratings)
        self.negs = self.initNegatives()

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating]"""
        print('normoalizing ratings')
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def getRated(self):
        rs = {}
        for i in self.ratings.groupby('userId'):
            rs[i[0]] = set(i[1].itemId)
        return rs

    def initNegatives(self):

        rs = {}
        for i in self.rated:
            rs[i] = list(self.item_pool.difference(self.rated[i]))

        return rs

    def _split_loo(self, ratings):
        """leave one out train/test split """
        print('get train test')
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        print('get neg sample')
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""

        """we need quick negtive sampling"""
        print('instance loader')
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings
        for row in train_ratings.itertuples():
            # print(row.Index)
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            negatives = random.sample(self.negs[int(row.userId)], num_negatives)
            users += [int(row.userId)] * num_negatives
            items += negatives
            ratings += [0.0] * num_negatives

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self, num=99):
        """create evaluate data"""
        test_ratings = self.test_ratings
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            # print('building evaluate data',row.Index)
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            negative_users.append([int(row.userId)] * num)
            negative_items.append(random.sample(self.negs[int(row.userId)], num))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]