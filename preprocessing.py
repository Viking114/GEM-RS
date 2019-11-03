from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import diags
from scipy.sparse import coo_matrix
import networkx as nx
import tqdm

def buildPreferenceNetwork(mat,sampleRate = 0.05,atLeast=5):
    idx = []
    adj = []
    weight = []

    coo = coo_matrix(mat)

    df = pd.DataFrame()

    df['idx'] = coo.row
    df['adj'] = coo.col
    df['weight'] = coo.data

    group = df.groupby('idx')
    count = 0

    for i in tqdm.tqdm(group):
        if i[0]!= count:
            print(i[0],'an no neighbor node is found')
            count += 1
        count += 1
        subdf = i[1]
        subdf = subdf.sort_values('weight', ascending=False)
        sort = subdf[subdf.weight != 1]
        end = int(len(sort) * sampleRate)
        if end+1 < atLeast:
            end = atLeast+1
        sort = sort[1:end]

        if len(sort) == 0:
            sort = subdf[1:atLeast+1]

        idx += [i[0]]*len(sort)
        adj += list(sort['adj'])
        weight += list(sort['weight'])

    return idx, adj, weight

def getAdjacentMat(idx,adj,weight):

    mat = csr_matrix((weight, (idx, adj)))
    return mat

def buildNetXfromSparseMat(mat):

    graph = nx.DiGraph()
    edgeCount = 0
    for i in tqdm.tqdm(range(0, mat.shape[0])):
        sr = pd.Series(mat[i].toarray()[0])
        sr = sr[sr!=0]
        idx = list(sr.index)
        weight = list(sr)

        for id,w in zip(idx,weight):
            graph.add_weighted_edges_from([(i,id,w)])
            edgeCount += 1

    print('totalEdge:',edgeCount)
    return graph

def getSimiGraph(df,uIdx='userId',iIdx='itemId',rtIdx='rating',Type='user',smplRate=1,atLeast=5,like=0,dislike=0,
                 interact_only=False):

    """
    :param smplRate: a value between 0-1, only keep top edges with high weights
    """
    userNum = df[uIdx].max() + 1
    itemNum = df[iIdx].max() + 1
    df['like'] = df[rtIdx].apply(lambda x:1 if x>=like else 0)
    df['dislike'] = df[rtIdx].apply(lambda x:1 if x<=dislike else 0)

    if interact_only:
        df[rtIdx] = 1

    df_like = df[df['like'] == 1]
    df_dislike = df[df['dislike']==1]
    mat1 = coo_matrix((df_like[rtIdx],(df_like[iIdx],df_like[uIdx])),shape=(itemNum,userNum))
    mat2 = coo_matrix((df_dislike[rtIdx],(df_dislike[iIdx],df_dislike[uIdx])),shape=(itemNum,userNum))

    mat = mat1 + mat2

    if Type == 'user':
        comat = mat.transpose() * mat
    else:
        comat = mat * mat.transpose()

    idx, adj, weight = buildPreferenceNetwork(comat,sampleRate=smplRate,atLeast=atLeast)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(idx,adj,weight))
    return graph