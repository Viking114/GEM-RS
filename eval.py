import math
import pandas as pd
import torch

"""
MetronATK is from https://github.com/yihong-chen/neural-collaborative-filtering
"""

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):

        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

def evaluate(model,evaluate_data, epoch_id,use_cuda):
    _metron = MetronAtK(top_k=10)
    with torch.no_grad():
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        if use_cuda is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
        test_scores = model.forward(test_users, test_items)
        negative_scores = model.forward(negative_users, negative_items)
        if use_cuda is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        _metron.subjects = [test_users.data.view(-1).tolist(),
                             test_items.data.view(-1).tolist(),
                             test_scores.data.view(-1).tolist(),
                             negative_users.data.view(-1).tolist(),
                             negative_items.data.view(-1).tolist(),
                             negative_scores.data.view(-1).tolist()]
    hit_ratio, ndcg = _metron.cal_hit_ratio(), _metron.cal_ndcg()
    print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
    return hit_ratio, ndcg