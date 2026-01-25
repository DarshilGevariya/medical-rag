from sentence_transformers import SentenceTransformer
import time

def embed(texts, model_name):
    model = SentenceTransformer(model_name)
    start = time.time()
    emb = model.encode(texts, show_progress_bar=True)
    elapsed = time.time() - start
    return emb, emb.shape[1], elapsed
