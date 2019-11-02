
from scipy.sparse import coo_matrix


def getPrefGraph_easy(df,like=0,dislike=0,uIdx='userId',iIdx='itemId',rtIdx='rating',Type='user',smplRate=1,atLeast=5,speicalForYelp=False):

    """
    make sure all user indices and item indices start from 0
    """
    userNum = df[uIdx].max() + 1
    itemNum = df[iIdx].max() + 1
    df['like'] = df[rtIdx].apply(lambda x:1 if x>=like else 0)
    df['dislike'] = df[rtIdx].apply(lambda x:1 if x<=dislike else 0)

    df_like = df[df['like'] == 1]
    df_dislike = df[df['dislike']==1]
    mat1 = coo_matrix((df_like[rtIdx],(df_like[iIdx],df_like[uIdx])),shape=(itemNum,userNum))
    mat2 = coo_matrix((df_dislike[rtIdx],(df_dislike[iIdx],df_dislike[uIdx])),shape=(itemNum,userNum))

    mat = mat1 + mat2

    if Type == 'user':
        mat = mat.transpose() * mat
    else:
        mat = mat * mat.transpose()
    print(mat.shape)
    comat = mat

    idx, adj, weight = buildPreferenceNetwork(comat,sampleRate=smplRate,atLeast=atLeast)
    df = normalizeGraph(idx, adj, weight, user=(Type == 'user'),yelp=speicalForYelp)
    graph = buildNetXfromSparseMat(df,)
    return graph