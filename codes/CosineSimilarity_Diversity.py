from sklearn.metrics.pairwise import cosine_similarity
import pandas

def club_similar_keywords(emb_mat, sim_score=0.9):
    """
    :param emb_mat: matrix having vectors with words as index
    :param sim_score: 0.9 by default
    :return: returns list of unique words from index after combining words which has similarity score of more than
    0.9
    """
    if len(emb_mat) == 0:
        return 'NA'
    xx = cosine_similarity(emb_mat)
    print(xx)
    final_keywords = set(emb_mat.index)
    N = len(emb_mat.index)
    dd = {}
    for i in range(N):
        for j in range(N):
            if (float(xx[i][j]) > sim_score) and (i != j):
                try:
                    dd[emb_mat.index[i]].append(emb_mat.index[j])
                except:
                    dd[emb_mat.index[i]] = []
                    dd[emb_mat.index[i]].append(emb_mat.index[j])
    removed_keywords = []
    for key in dd:
        for val in dd[key]:
            if key not in removed_keywords:
                removed_keywords += dd[key]
                try:
                    final_keywords.remove(val)
                except:
                    pass
    return final_keywords

print(club_similar_keywords(pandas.DataFrame(index=['Product', 'Jungle', 'Good Product', 'Lightweight'])))