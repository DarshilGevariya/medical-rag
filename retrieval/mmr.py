def mmr(docs, scores, lambda_=0.5):
    selected = []
    while docs:
        best = max(
            docs,
            key=lambda d: lambda_ * scores[d] -
            (1 - lambda_) * sum(scores.get(s, 0) for s in selected)
        )
        selected.append(best)
        docs.remove(best)
    return selected
