from rouge_score import rouge_scorer
import re


def rouge_l(prediction, reference):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score["rougeL"].fmeasure


def citation_coverage(text):
    """
    Fraction of sentences that contain at least one citation like [c1], [c2]
    """
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    cited = [s for s in sentences if "[" in s and "]" in s]
    return len(cited) / len(sentences)
