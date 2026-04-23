## Project Overview

This project implements a Medical RAG (Retrieval-Augmented Generation) system that combines semantic document retrieval with LLM-based answer generation to provide accurate, context-grounded medical responses.

## What the Project Does
Accepts natural language medical queries from users
Retrieves relevant medical information from a document corpus
Uses an LLM to generate answers based only on retrieved context
Ensures responses are factually grounded and less prone to hallucination
## Implemented

**1. Document Ingestion Pipeline**
Loaded medical dataset(**MedQuad** used here)
Performed text extraction and preprocessing
Applied chunking (splitting into smaller meaningful segments) for better retrieval

**2. Embedding Generation**
Converted text chunks into dense vector embeddings
Embedding models used: **all-MiniLM-L6-v2, all-mpnet-base-v2**
Ensured semantic similarity capture for better search

**3. Vector Database Integration**
Stored embeddings in a vector database (used FAISS here)
Enabled efficient similarity search
Configured top-k retrieval (e.g., k = 1,5,10)

**4. Retrieval Mechanism**
Converted user query into embedding
Performed cosine similarity search
Retrieved top-k most relevant document chunks

**5. RAG Pipeline**
Combined:
Retriever (vector search)
Generator (used **flan-t5-base trasnformer** here)
Passed retrieved context into LLM prompt
Generated context-aware and grounded answers

**6. Prompt Engineering**
Designed prompts to:
Restrict model to use only retrieved context
Improve answer relevance and factual accuracy

**7. End-to-End Query System**
Built a pipeline that:
Takes user query
Retrieves relevant chunks
Generates final answer
Ensured low-latency response flow

**8. Evaluation** 
Used metrics like:
Recall@k (e.g., Recall@5) for retrieval performance
Cosine similarity for ranking relevance
Observed improved answer grounding and reduced hallucination

# Key Outcomes

✅ Reduced hallucination compared to standalone LLM

✅ Improved answer accuracy using real medical data

✅ Built a scalable and modular RAG pipeline

✅ Enabled context-aware medical query handling

## How to run

### Setup
pip install -r requirements.txt

### Parse data
python -m data.parse_medquad

### Build index
python -m indexing.build_index

### Run evaluation
python -m evaluation.run_experiments

### Run generation demo
python -m generation.generate

### Run tests
python -m pytest tests


# Frequently asked Questions related to project

## 1. Chunking
**What chunking strategy did you implement and why? What failure modes did you observe?**

I have implemented fixed-size character-based chunking with overlap. Each document was split into chunks of **256 characters with an overlap of 50 characters**. This in balancing semantic completeness and retrieval granularity while remaining independent of tokenization and model architecture. The overlap ensures that important information at chunk boundaries is not lost.

**Failure modes observed:**  
When chunk sizes were too large, chunks often contained multiple unrelated facts, which diluted relevance and reduced retrieval precision. When chunks were too small, important contextual information was spreaded across many chunks, leading to incomplete or weakly grounded answers. Excessive overlap increased redundancy and retrieval cost without providing proportional gains in recall.


## 2. Embeddings
**Why did one embedding model win or lose in your experiments?**

I have taken two embedding models: **all-MiniLM-L6-v2** and **all-mpnet-base-v2**. MiniLM performed significantly faster and was more CPU-efficient, making it suitable for large-scale experiments and repeated evaluations. MPNet consistently achieved higher recall and MRR scores due to its richer semantic representations, but at the cost of increased computation time and memory usage. The observed trade-off was between efficiency (MiniLM) and retrieval quality (MPNet).


## 3. Retrieval
**What changed when you varied `top_k`? Did you try MMR or hybrid retrieval, and what trade-offs did they introduce?**

Increasing `top_k` generally improved recall by retrieving more potentially relevant chunks, but it also introduced additional noise that reduced answer precision. I have implemented **MMR (Maximal Marginal Relevance)** to reduce redundancy among retrieved chunks, which improved diversity and grounding. I also tested **hybrid retrieval (dense + BM25)**, which improved recall by combining semantic and lexical matching. However, hybrid retrieval required careful tuning, as it sometimes introduced lexically similar but semantically weak chunks.


## 4. Evaluation
**Which metric best matched “real” quality, and which metric was misleading?**

Among the evaluated metrics, **MRR@10** best matched perceived answer quality because it rewards placing relevant context early in the ranked retrieval results. **Recall@10**, while useful, was sometimes misleading: high recall did not always translate into better generated answers if relevant chunks appeared late in the ranking. ROUGE-L showed moderate correlation with answer quality but was insensitive to factual grounding and citation correctness.


## 5. Faithfulness
**How did you reduce hallucinations and improve citation quality?**

Hallucinations were reduced by explicitly restricting the generation prompt to retrieved context only. Additionally, citation presence was enforced post-generation to prevent citation drop during paraphrasing. I used symbolic citations (`[c_i]`) that directly reference retrieved chunks, enabling traceability. Citation quality was quantified using a **citation coverage** metric, which measures the fraction of answer sentences supported by citations.


## 6. Future Improvements
**If you had two more weeks, what would you improve first?**

With additional time, I would prioritize adding a **cross-encoder reranker** to improve retrieval precision. Next, I would explore domain-specific embedding fine-tuning for medical text. Further improvements would include prompt optimization for stricter faithfulness, enhanced evaluation of grounding quality, and potentially supervised fine-tuning of the generation model.
