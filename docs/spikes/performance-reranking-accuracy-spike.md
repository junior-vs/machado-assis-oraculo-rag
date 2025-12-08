---
title: Document Reranking and Retrieval Accuracy Optimization
category: Performance & Scalability
status: "🟡 In Progress"
priority: Medium
timebox: 2 weeks
created: 2025-12-05
updated: 2025-12-05
owner: ML Team
tags: ["technical-spike", "performance", "accuracy", "rag", "retrieval"]
---

# Document Reranking and Retrieval Accuracy Optimization

## Summary

**Spike Objective:** Research and evaluate document reranking strategies to improve retrieval accuracy beyond basic semantic similarity, including hybrid search and cross-encoder reranking.

**Why This Matters:** Current RAG system relies on cosine similarity for document retrieval. For complex queries or nuanced literary content like Machado de Assis's work, semantic similarity alone may miss relevant passages. Reranking improves answer quality and reduces hallucinations.

**Timebox:** 2 weeks for accuracy evaluation, reranking algorithm testing, and performance benchmarking

**Decision Deadline:** Before production deployment to ensure high answer quality

## Research Question(s)

**Primary Question:** How can we improve document retrieval accuracy for literary Q&A without significantly increasing latency or infrastructure complexity?

**Secondary Questions:**

- How much does reranking improve answer quality for literary texts?
- What's the latency impact of adding reranking to retrieval pipeline?
- How do different reranking strategies compare (BM25, cross-encoders, LLM-based)?
- Should we implement hybrid search combining semantic + keyword matching?
- Can we use LLM's generation quality as a metric for retrieval evaluation?

## Investigation Plan

### Research Tasks

- [ ] Establish baseline retrieval quality with current semantic search
- [ ] Build evaluation dataset with example queries and expected documents
- [ ] Implement BM25 (keyword) search as fallback/hybrid approach
- [ ] Test cross-encoder reranking models (e.g., ms-marco-TinyBERT)
- [ ] Benchmark reranking latency and cost impact
- [ ] Evaluate RAG-as-Judge for retrieval quality assessment
- [ ] Design metrics for literary content accuracy
- [ ] Create performance/accuracy trade-off analysis

### Success Criteria

**This spike is complete when:**

- [ ] Baseline retrieval accuracy established (10+ test queries)
- [ ] Reranking strategies tested and compared
- [ ] Latency impact quantified for each approach
- [ ] Evaluation metrics defined and measured
- [ ] Recommendation made for production implementation
- [ ] POC prototype completed for selected strategy

## Technical Context

**Related Components:**
- `src/infrastructure/vector_store.py` - Document retrieval
- `src/use_cases/nodes.py` - Grading logic
- `src/use_cases/graph.py` - Query transformation

**Dependencies:**
- LangChain >=0.2.0
- Sentence Transformers (reranking models)
- FAISS (current retrieval)
- Optional: scikit-learn (BM25), Elasticsearch (hybrid search)

**Constraints:**
- Query latency must remain <2 seconds total
- Must work offline with downloaded models
- No additional API costs for reranking
- Model size must fit in memory (< 2GB)

## Research Findings

### Investigation Results

**Current Retrieval Analysis:**

**Mechanism:**

```python
# Current: Pure semantic similarity
retriever = vectorstore.as_retriever(search_type="similarity", k=3)
docs = retriever.invoke(question)
# Returns top-3 documents by cosine similarity to question embedding
```

**Strengths:**

✅ Fast: ~5-10ms query time
✅ Simple: No additional configuration
✅ Effective for straightforward queries
✅ Works well for well-indexed content

**Limitations:**

❌ May miss paraphrased content
❌ Doesn't understand query intent deeply
❌ Keyword synonyms not captured
❌ Phrase matching ignored
❌ Literary allusions may not match directly

**Retrieval Challenges for Literary Content:**

1. **Multiple valid answers** - Same concept discussed in different passages
2. **Paraphrasing** - Same idea expressed differently
3. **Indirect references** - Themes woven throughout without direct mention
4. **Context dependency** - Meaning depends on literary context

### Prototype/Testing Notes

**Test Scenario 1: Query Mismatch Analysis**
```
Query: "Who is the protagonist of Memórias Póstumas?"
Current result: Returns document mentioning "Brás Cubas" by name
Assessment: ✅ Works when character name appears

Query: "Tell me about the dead narrator in this book"
Current result: May not return "Memórias Póstumas" section if not directly stated
Assessment: ⚠️ Indirect references less effective
```

**Test Scenario 2: Semantic Similarity vs Relevance**
```
Query: "What is the book's main style?"
Current retrieval:
- Doc 1: "Machado's literary style..." (high similarity, high relevance) ✅
- Doc 2: "The author wrote in nineteenth century..." (medium similarity, medium relevance)
- Doc 3: "Brazilian literature examples..." (low similarity, low relevance)

Assessment: ✅ Ranking is mostly correct for style questions
```

### Reranking Strategy Evaluation

**Option 1: BM25 Hybrid Search**

```
Implementation: Combine semantic + keyword matching
Formula: 0.3 * BM25(query, doc) + 0.7 * cosine_similarity(query_emb, doc_emb)

Pros:
✅ Captures exact keyword matches
✅ Handles acronyms and named entities
✅ Relatively fast
✅ No additional models needed

Cons:
❌ Weights arbitrary (0.3/0.7)
❌ Requires full-text indexing
❌ Doesn't understand semantic relationships
```

**Option 2: Cross-Encoder Reranking**

```
Implementation: Use dedicated reranking model on top-k results
Pipeline: semantic_search(k=10) → cross_encoder_score() → rerank(k=3)

Models:
- ms-marco-TinyBERT: ~130MB, ~5ms per document
- BGE-reranker-base: ~600MB, ~10ms per document
- UAE-Large-V1-Instruct: ~1GB, ~20ms per document

Pros:
✅ Significantly improves relevance ranking
✅ Understands query-document relationships
✅ No hyperparameter tuning needed
✅ Works offline

Cons:
❌ Adds latency (5-20ms per document)
❌ Model download and memory cost
❌ Increases complexity
```

**Option 3: LLM-as-Reranker**

```
Implementation: Use LLM to score document relevance
Query: "How relevant is this passage to '{question}'?"
Score: 1-5 relevance rating from LLM

Pros:
✅ State-of-the-art accuracy
✅ Understands complex relevance
✅ Can explain relevance decisions

Cons:
❌ Slow: ~500ms per document
❌ Expensive: ~$0.001 per document scored
❌ Impractical for 10+ document reranking
```

**Latency Impact Analysis:**

```
Current pipeline (semantic only):
retrieve(k=3): 5-10ms
grade_documents: 100-500ms (LLM call)
Total: 105-510ms ✅

With BM25 hybrid:
semantic_search(k=3): 5-10ms
bm25_score(k=3): 5-10ms
merge_scores: <1ms
Total: 10-21ms (same tier) ✅

With cross-encoder reranking:
semantic_search(k=10): 10-20ms
cross_encoder_score(k=10): 50-100ms
rerank and return top-3: <1ms
Total: 60-120ms (⚠️ 12x slower than current) ⚠️

With LLM reranking:
semantic_search(k=10): 10-20ms
llm_score(k=10): 5000-10000ms
Total: 5010-10020ms (❌ too slow)
```

### Recommended Strategy

**Hybrid Approach with Cross-Encoder:**

```
Pipeline:
1. semantic_search(k=10)  [10-20ms]
2. cross_encoder_rerank(k=10) [50-100ms]
3. return top-3 [<1ms]
4. grade_documents (existing) [100-500ms]
Total: 160-620ms (acceptable, current is 105-510ms)
```

**Implementation Path:**

- **Phase 1:** Add BM25 hybrid search (low cost, immediate benefit)
- **Phase 2:** Evaluate cross-encoder reranking (measurable improvement)
- **Phase 3:** Optimize LLM-as-judge for final relevance scoring

## Decision

### Recommendation (Tentative)

**Implement Staged Reranking Strategy**

**Phase 1 (Immediate):** Hybrid Search with BM25
- Add keyword matching to semantic search
- Simple to implement, no additional dependencies
- Improve coverage of exact phrase matches

**Phase 2 (Production):** Cross-Encoder Reranking
- Integrate TinyBERT cross-encoder for top-10 documents
- ~100ms added latency acceptable for better quality
- Significantly improves relevance ranking

**Phase 3 (Optional):** LLM Reranking for Critical Queries
- Use GPT-4 or Claude as final judge
- Only for queries where confidence is low
- Accept higher latency for quality guarantee

### Rationale

- **Quick Win:** BM25 hybrid low effort, immediate improvement
- **Quality Leap:** Cross-encoder proves reranking value
- **Cost Effective:** Phase 1 & 2 don't require external APIs
- **Gradual:** Can implement iteratively without major refactoring
- **Measurable:** Improvements can be validated with test queries

### Implementation Notes

**For Hybrid Search:**

```python
# src/infrastructure/vector_store.py
from rank_bm25 import BM25Okapi

def hybrid_retrieve(question: str, k: int = 3) -> List[Document]:
    # Semantic search
    semantic_results = vector_search(question, k=10)
    
    # BM25 search
    bm25_results = bm25_search(question, k=10)
    
    # Combine and rerank
    combined = merge_results(semantic_results, bm25_results)
    return combined[:k]
```

**For Cross-Encoder Reranking:**

```python
# Requires: pip install sentence-transformers

from sentence_transformers import CrossEncoder

reranker = CrossEncoder('ms-marco-TinyBERT-L-12')

def rerank_documents(question: str, documents: List[Document]) -> List[Document]:
    scores = reranker.predict(
        [[question, doc.page_content] for doc in documents]
    )
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked]
```

### Follow-up Actions

- [ ] Create evaluation dataset with 20+ test queries
- [ ] Implement BM25 hybrid search POC
- [ ] Test cross-encoder reranking with TinyBERT
- [ ] Measure accuracy improvements with evaluation metrics
- [ ] Benchmark latency impact on full pipeline
- [ ] Document configuration for reranking parameters
- [ ] Create A/B testing framework for comparing strategies
- [ ] Plan Phase 2 implementation for production

## Status History

| Date       | Status         | Notes                                          |
| ---------- | -------------- | ---------------------------------------------- |
| 2025-12-05 | 🟡 In Progress | Initial research and strategy evaluation       |

---

_Last updated: 2025-12-05 by ML Team_
_Reference Implementation: src/infrastructure/vector_store.py_
_Reranking Research: https://huggingface.co/cross-encoders_
_RAG Evaluation: https://github.com/run-llama/llama-index/tree/main/llama-index-core/llama_index/evaluation_
