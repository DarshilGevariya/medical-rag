from indexing.chunking import chunk_text
from evaluation.retrieval_eval import recall_at_k, mrr_at_k
from evaluation.generation_eval import citation_coverage


def test_chunking_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=5)

    # overlap means characters repeat
    assert chunks[1].startswith("fghij")


def test_recall_at_k():
    retrieved = [1, 2, 3, 4]
    relevant = [3]

    assert recall_at_k(retrieved, relevant, k=3) == 1
    assert recall_at_k(retrieved, relevant, k=1) == 0


def test_mrr_at_k():
    retrieved = [5, 4, 3, 2]
    relevant = [3]

    assert mrr_at_k(retrieved, relevant, k=4) == 1 / 3


def test_citation_coverage():
    text = "This is a sentence [c1]. This is another sentence."
    coverage = citation_coverage(text)

    assert coverage == 0.5
