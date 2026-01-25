from rank_bm25 import BM25Okapi

def hybrid(query, docs, dense_scores, alpha=0.5):
    bm25 = BM25Okapi([d.split() for d in docs])
    bm25_scores = bm25.get_scores(query.split())
    return alpha * dense_scores + (1 - alpha) * bm25_scores
