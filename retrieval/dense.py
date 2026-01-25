def retrieve(index, query_emb, top_k):
    scores, ids = index.search(query_emb, top_k)
    return ids[0]
