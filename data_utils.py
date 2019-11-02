import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from os import path

filePath = path.dirname(__file__)

def kCore(df,k=5):

    graph = nx.Graph()
    df_ = df.copy()
    df_['userId'] = df_['userId'].apply(lambda x:'u'+str(x))
    df_['itemId'] = df_['itemId'].apply(lambda x:'i'+str(x))
    graph.add_edges_from([i for i in zip(list(df_['userId']),list(df_['itemId']))])
    subGraph = nx.algorithms.k_core(graph,k=k)

    nodeList = list(subGraph.nodes)
    user = 0
    for i in nodeList:
        if i[0] == 'u':
            user += 1

    print('userNum:',user)
    print('itemNum',len(nodeList)-user)
    users = []
    items = []

    for i in subGraph.nodes:
        if i[0] == 'i':
            i = i.replace('i','')
            i = int(i) if str.isdigit(i) else i
            items.append(i)
        elif i[0] == 'u':
            i = i.replace('u', '')
            i = int(i) if str.isdigit(i) else i
            users.append(i)

    df_ = df[df.userId.isin(users)]
    df_ = df_[df_.itemId.isin(items)]

    return df_,df

def kaggleMovieProcessImdbId(ids):
    rs = []
    for i in ids:
        s = str(i)
        s = 'tt' + '0'*(7-len(s)) + s
        rs.append(s)
    return rs

def loadKaggleMovieSmallLinks():
    links = pd.read_csv(filePath+r'\dataset\Kaggle-MLextend\links_small.csv')
    links.columns = ['movieId', 'imdb_id', 'tmdbId']
    links['imdb_id'] = kaggleMovieProcessImdbId(list(links['imdb_id']))
    return links

def relabelUserAndItem(df,uIdx='userId',iIdx='itemId'):

    uEncoder = LabelEncoder()
    iEncoder = LabelEncoder()
    originUidx = df[uIdx]
    originIidx = df[iIdx]
    df[uIdx] = uEncoder.fit_transform(list(df[uIdx]))
    df[iIdx] = iEncoder.fit_transform(list(df[iIdx]))
    df['originUId'] = originUidx
    df['originIid'] = originIidx

    return df

def loadKaggleMovieSmall():

    links = loadKaggleMovieSmallLinks()
    meta = pd.read_csv(filePath+r'\dataset\Kaggle-MLextend\movies_metadata.csv')
    rts = pd.read_csv(filePath+r'\dataset\Kaggle-MLextend\ratings_small.csv')
    df = links.merge(meta, on=['imdb_id'])
    rts = rts[rts.movieId.isin(df.movieId)]
    m = rts.groupby('movieId').count()
    rts = rts[rts.movieId.isin(m[m.userId >= 2].index)]
    rts.columns = ['userId', 'itemId', 'rating', 'timestamp']
    df = df[df['movieId'].isin(set(list(rts.itemId)))]
    rts = relabelUserAndItem(rts)
    metaData = df
    return rts,metaData



if __name__ == '__main__':
    rt,meta = loadKaggleMovieSmall()