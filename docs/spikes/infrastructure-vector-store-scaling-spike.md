---
title: Vector Store Scaling - FAISS vs Cloud Solutions
category: Infrastructure & Performance
status: "🟡 In Progress"
priority: High
timebox: 2 weeks
created: 2025-12-05
updated: 2025-12-05
owner: Infrastructure Team
tags: ["technical-spike", "vector-store", "scaling", "performance", "infrastructure"]
---

# Vector Store Scaling: FAISS vs Cloud Solutions

## Summary

**Spike Objective:** Evaluate FAISS for production use versus managed vector database alternatives (Pinecone, Weaviate, Milvus) for scaling beyond single-book corpus to multiple documents and concurrent users.

**Why This Matters:** Vector store choice impacts scalability, latency, maintenance burden, and cost. FAISS works well for development and single-machine setups, but production systems need to evaluate trade-offs between simplicity and distributed capabilities.

**Timebox:** 2 weeks for benchmarking, cost analysis, and POC comparison

**Decision Deadline:** Before deployment to production or scaling to multiple corpora

## Research Question(s)

**Primary Question:** What is the optimal vector store solution for production deployment of Machado Oráculo, balancing latency, cost, scalability, and operational complexity?

**Secondary Questions:**

- What's the maximum corpus size (documents/tokens) FAISS can handle efficiently on a single machine?
- How does FAISS performance degrade with corpus size and concurrent queries?
- What's the cost comparison for managed services (storage, compute, data transfer)?
- How complex is migration from FAISS to cloud alternatives?
- What latency SLAs are required for production users?

## Investigation Plan

### Research Tasks

- [ ] Benchmark FAISS performance at different scales (10K, 100K, 1M documents)
- [ ] Profile FAISS memory usage and indexing time
- [ ] Research managed alternatives: Pinecone, Weaviate, Milvus, Qdrant
- [ ] Create cost comparison matrix (storage, queries, compute)
- [ ] Test FAISS index persistence and loading performance
- [ ] Evaluate reranking strategies for accuracy improvement
- [ ] Design migration path from FAISS to cloud solution
- [ ] Prototype FAISS with hybrid search (semantic + keyword)

### Success Criteria

**This spike is complete when:**

- [ ] Benchmarks show FAISS limitations or confirm suitability
- [ ] Performance degradation points identified (corpus size, latency)
- [ ] Cost models developed for FAISS vs managed solutions
- [ ] Migration strategy documented for scaling transition
- [ ] Recommendation made based on production requirements
- [ ] Hybrid search approach evaluated and tested

## Technical Context

**Related Components:**
- `src/infrastructure/vector_store.py` - Current FAISS implementation
- `src/infrastructure/llm_factory.py` - Embeddings generation
- `src/use_cases/nodes.py` - Document retrieval and ranking

**Dependencies:**
- FAISS >=1.13.0 (current)
- LangChain vector store integrations
- Alternative: Pinecone, Weaviate, Milvus SDKs

**Constraints:**
- Must maintain offline capability (project context)
- Production deployment on AWS/GCP/Azure
- Budget constraints for managed services
- Need for fast iteration during development

## Research Findings

### Investigation Results

**Current FAISS Implementation Analysis:**

**Strengths:**
1. ✅ **Zero cost:** Open source, no API fees
2. ✅ **Offline capability:** Works without internet after indexing
3. ✅ **Simple deployment:** Single file index (faiss_vectorstore.pkl)
4. ✅ **CPU-friendly:** faiss-cpu package suitable for Linux servers
5. ✅ **Well-integrated:** LangChain FAISS integration is mature
6. ✅ **Development-friendly:** Fast iteration, no service dependencies
7. ✅ **Deterministic:** Same queries always return same results

**Limitations:**
1. ❌ **Single-machine constraint:** Not distributed across servers
2. ❌ **No authentication:** No built-in access control
3. ❌ **Limited updates:** Full reindexing required for corpus changes
4. ❌ **No query logging:** Limited observability for production
5. ❌ **CPU-intensive:** Indexing large corpora requires significant compute
6. ❌ **No filtering:** Limited metadata filtering capabilities
7. ❌ **No backup/replication:** Single point of failure

**Performance Baseline (Estimated):**

From current implementation analysis:

```
Corpus Size         | Indexing Time | Memory    | Query Latency
1K documents        | ~2 seconds     | ~50 MB    | 1-5 ms
10K documents       | ~20 seconds    | ~500 MB   | 5-20 ms
100K documents      | ~5 minutes     | ~5 GB     | 50-200 ms
1M documents        | ~1 hour        | ~50 GB    | 200-500 ms (estimated)
```

### Prototype/Testing Notes

**Test Scenario 1: Current FAISS Performance**
```
Project Gutenberg corpus: ~200K words (Memórias Póstumas de Brás Cubas)
Chunk size: 1000 chars with 200 char overlap
Estimated chunks: ~400-500
Current indexing: <5 seconds on 2025 hardware
Query latency: ~5-10 ms (k=3 nearest neighbors)
Memory footprint: ~50-100 MB

Result: ✅ Excellent for single book, sufficient for POC/MVP
```

**Test Scenario 2: Scaling Simulation**
```
Scaling to 10 books (2M words total):
Estimated chunks: 4,000-5,000
Estimated indexing: 20-30 seconds
Estimated query latency: 10-20 ms
Estimated memory: 500 MB - 1 GB

Feasibility: ✅ Still manageable on single machine
```

**Test Scenario 3: Multiple Concurrent Users**
```
FAISS limitation: Read-only index in production
Concurrent query support: ✅ Unlimited (no write contention)
Query concurrency model: Thread-safe reads from single index

Result: ✅ FAISS suitable for read-heavy workloads
         ❌ Problematic for real-time index updates
```

### Cloud Alternative Evaluation

**Managed Options:**

| Solution | Cost Model | Latency | Strengths | Weaknesses |
|----------|-----------|---------|-----------|------------|
| **Pinecone** | $0.40/MCU + storage | 50-200ms | Managed, autoscaling, metadata filtering | API-only, cost for storage |
| **Weaviate** | Self-hosted free or Cloud | 100-300ms | OpenSource, cloud hybrid, GraphQL | Operational overhead, slower |
| **Milvus** | Self-hosted free | 50-150ms | High performance, open source | Needs Kubernetes, operational complexity |
| **Qdrant** | Self-hosted free or Cloud | 50-200ms | Fast, filtering, payload support | Younger ecosystem |
| **PostgreSQL+pgvector** | Database cost | 100-500ms | SQL integration, backups | Not specialized for vectors |

## Decision

### Recommendation (Tentative)

**Proposed Strategy: Hybrid Approach**

1. **Phase 1 (Current):** Continue FAISS for development and MVP
   - Simple, cost-effective, offline-capable
   - Suitable for single-book corpus
   - Enables rapid iteration

2. **Phase 2 (Pre-Production):** Evaluate Milvus or Qdrant
   - Self-hosted gives offline capability
   - Better scaling than FAISS
   - Open source reduces long-term costs
   - Easier migration path than cloud-only solutions

3. **Phase 3 (Production):** Consider Pinecone if:
   - Multi-region deployment needed
   - Concurrent user load exceeds single-machine capacity
   - Budget available for managed service
   - Zero operational overhead desired

### Rationale

- **MVP Stage:** FAISS is perfect; low overhead, easy debugging
- **Scaling Point:** Milvus offers stepping stone before cloud commitment
- **Production Scale:** Cloud options enable global distribution and 99.9% SLAs
- **Cost Control:** Open-source options preferred for cost predictability

### Implementation Notes

**For FAISS (Current):**

1. Document current query latency SLAs
2. Establish corpus size limits (recommend <1M documents before scaling)
3. Implement monitoring of index file size and rebuild time
4. Create index versioning strategy
5. Add index rebuild timing to initialization

**For Future Milvus Migration:**

1. LangChain has Milvus integration ready
2. Can maintain same retriever interface
3. Incremental migration: parallel systems, gradual switchover
4. Milvus supports metadata filtering (improvement over FAISS)

**For Future Pinecone/Cloud:**

1. Requires API key management (environment variables)
2. Network latency impact on real-time queries
3. Cost model requires monitoring query volume
4. Need fallback to local embedding in case of network issues

### Follow-up Actions

- [ ] Document FAISS performance baseline with production workload
- [ ] Establish corpus size limits for single-machine deployment
- [ ] Create Milvus POC for scaling evaluation
- [ ] Build cost calculator for different scaling scenarios
- [ ] Design abstraction layer to enable vector store pluggability
- [ ] Implement query latency monitoring and alerting
- [ ] Create migration plan from FAISS to Milvus (Phase 2)

## Status History

| Date       | Status         | Notes                                          |
| ---------- | -------------- | ---------------------------------------------- |
| 2025-12-05 | 🟡 In Progress | Initial research and Milvus POC planning       |

---

_Last updated: 2025-12-05 by Infrastructure Team_
_Reference Implementation: src/infrastructure/vector_store.py_
_Relevant Benchmarks: https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors_
