def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    return int(any(r in retrieved_k for r in relevant_ids))


def mrr_at_k(retrieved_ids, relevant_ids, k):
    for i, r in enumerate(retrieved_ids[:k], start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0
