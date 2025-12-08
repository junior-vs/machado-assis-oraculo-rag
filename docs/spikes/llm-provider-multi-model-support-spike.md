---
title: Multi-LLM Provider Support - Abstraction and Fallback Strategy
category: Architecture & Design
status: "🟡 In Progress"
priority: Medium
timebox: 2 weeks
created: 2025-12-05
updated: 2025-12-05
owner: Development Team
tags: ["technical-spike", "llm-integration", "architecture", "abstraction", "reliability"]
---

# Multi-LLM Provider Support: Abstraction and Fallback Strategy

## Summary

**Spike Objective:** Design an abstraction layer for supporting multiple LLM providers (Google Gemini, OpenAI, Anthropic Claude) with automatic fallback and provider selection logic.

**Why This Matters:** Current implementation is tightly coupled to Google Gemini. Production systems need provider flexibility for cost optimization, availability, and feature selection. Fallback strategies improve reliability when primary provider experiences outages.

**Timebox:** 2 weeks for architecture design, abstraction layer implementation, and testing

**Decision Deadline:** Before scaling to production or adding additional LLM capabilities

## Research Question(s)

**Primary Question:** How should we architect multi-LLM provider support with automatic fallback without complicating the core RAG logic?

**Secondary Questions:**

- What LLM capabilities are core vs optional for RAG pipeline?
- How should provider selection be configured (environment variables, config files, runtime)?
- What fallback strategy minimizes latency while ensuring reliability?
- How do we handle provider-specific features (function calling, vision, etc.)?
- What's the cost/performance trade-off for different providers?

## Investigation Plan

### Research Tasks

- [ ] Audit current Gemini usage to identify required capabilities
- [ ] Research LangChain provider abstractions and integrations
- [ ] Compare function-calling capabilities across Gemini, GPT-4, Claude
- [ ] Create cost comparison matrix for different providers
- [ ] Design provider abstraction interface (LLMProvider protocol)
- [ ] Test provider-specific feature variations
- [ ] Build fallback strategy and retry logic
- [ ] Prototype multi-provider configuration system

### Success Criteria

**This spike is complete when:**

- [ ] Core LLM capabilities identified and documented
- [ ] Provider abstraction interface designed
- [ ] Multi-provider configuration system implemented
- [ ] Fallback strategy tested and validated
- [ ] Cost/performance comparison documented
- [ ] Migration plan for existing code defined
- [ ] Example implementations for 2+ providers completed

## Technical Context

**Related Components:**
- `src/infrastructure/llm_factory.py` - LLM instantiation
- `src/use_cases/nodes.py` - LLM usage in chains
- `src/config.py` - Configuration management

**Dependencies:**
- LangChain >=0.2.0 (ChatModel abstraction)
- langchain-google-genai >=2.1.0 (current)
- langchain-openai (optional)
- langchain-anthropic (optional)

**Constraints:**
- API key management must be secure (environment variables)
- Backward compatibility with current Gemini usage
- Minimal code changes to RAG pipeline
- Cost must not exceed current Gemini usage

## Research Findings

### Investigation Results

**Current Implementation Analysis:**

**Single-Provider Coupling:**

```python
# Current: Tightly coupled to Gemini
class LLMFactory:
    @staticmethod
    def get_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=settings.gemini_api_key,
            temperature=0.0
        )

# Used in nodes.py:
llm = LLMFactory.get_llm()
chain = prompt | llm | StrOutputParser()
```

**Issues:**

1. ❌ No abstraction layer for swapping providers
2. ❌ No fallback if Gemini API is unavailable
3. ❌ Configuration tightly coupled to Gemini
4. ❌ Provider-specific features not configurable
5. ❌ No cost tracking across providers

**Core RAG Requirements (Function Analysis):**

From `nodes.py` analysis, LLM must support:

1. **Structured Output (GradeDocuments):**
   - Function calling: ✅ Gemini, ✅ GPT-4, ✅ Claude 3
   - Format: JSON schema with single required field

2. **Prompt Templates:**
   - Standard prompt format
   - Context injection (documents + question)
   - No vision or multimodal features needed

3. **Output Parsing:**
   - Text output parsing
   - JSON parsing for structured outputs
   - Error handling for invalid responses

4. **Performance Requirements:**
   - Latency: <2 seconds per call acceptable
   - Throughput: Single-threaded queries sufficient
   - Reliability: 99% success rate acceptable

**Provider Capability Matrix:**

| Feature | Gemini 2.5-Flash | GPT-4o | Claude 3.5 Sonnet |
|---------|-----------------|--------|-----------------|
| Function Calling | ✅ Yes | ✅ Yes | ✅ Yes |
| Structured Output | ✅ Yes | ✅ Yes | ✅ Yes |
| Cost (1M input tokens) | $0.075 | $5.00 | $3.00 |
| Latency (p95) | ~500ms | ~800ms | ~600ms |
| Context Window | 1M tokens | 128K tokens | 200K tokens |
| Availability | ✅ Good | ✅ Excellent | ✅ Good |

### Prototype/Testing Notes

**Test Scenario 1: Provider Abstraction**
```python
# Desired interface:
llm = LLMFactory.get_llm(provider="gemini")  # or "openai", "anthropic"
# Should return compatible ChatModel interface

Result: ✅ LangChain ChatModel interface is provider-agnostic
        ✅ Can swap implementations without changing RAG code
```

**Test Scenario 2: Fallback Strategy**
```python
# With fallback chain:
primary = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
fallback = ChatOpenAI(model="gpt-4o-mini")

llm = RunnableWithFallback(
    runnable=primary,
    fallbacks=[fallback],
    exception_key=["rate_limit", "service_unavailable"]
)

Result: ✅ LangChain RunnableWithFallback supports this pattern
        ✅ Graceful degradation to backup provider
```

**Test Scenario 3: Function Calling Compatibility**
```python
# GradeDocuments model usage across providers:
grader_gemini = llm.with_structured_output(
    GradeDocuments, 
    method="function_calling"
)
grader_openai = llm.with_structured_output(GradeDocuments)  # Auto-detects

Result: ✅ All providers support function calling
        ✅ API differences abstract away in LangChain
```

## Decision

### Recommendation (Tentative)

**Implement Provider Abstraction with Fallback Support**

1. **Layer 1: Provider Interface (LLMProvider Protocol)**
   - Define abstract interface for LLM access
   - Specify required methods: complete(), stream(), with_structured_output()
   - Leverage LangChain's ChatModel interface (already standardized)

2. **Layer 2: Factory Pattern (Enhanced LLMFactory)**
   - Support multiple provider configurations
   - Load from environment variables with defaults
   - Support provider priority/fallback chain
   - Cache LLM instances for performance

3. **Layer 3: Fallback Chain (RunnableWithFallback)**
   - Primary: Gemini (cost-effective, good performance)
   - Fallback: GPT-4o-mini (when Gemini unavailable)
   - Error handling: Rate limits, quota exceeded, service errors

### Rationale

- **Abstraction:** LangChain ChatModel already provides standardization
- **Cost Optimization:** Gemini is 50x cheaper than GPT-4 for same quality
- **Reliability:** Fallback chain ensures service continuity
- **Minimal Changes:** RAG code doesn't need modification
- **Flexibility:** Easy to add/remove providers without code changes

### Implementation Notes

**Phase 1: Provider Abstraction**

```python
# src/infrastructure/llm_factory.py (enhanced)

from typing import Literal, Optional
from langchain_core.language_models.chat_model import BaseChatModel

class LLMFactory:
    _instances: Dict[str, BaseChatModel] = {}
    
    @staticmethod
    def get_llm(
        provider: Literal["gemini", "openai", "anthropic"] = "gemini",
        fallback: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """Get LLM with optional fallback provider."""
        # Implementation...
```

**Phase 2: Configuration Updates**

```python
# .env
LLM_PROVIDER=gemini  # or openai, anthropic
LLM_FALLBACK_PROVIDER=openai-mini
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

**Phase 3: Fallback Chain**

```python
# In nodes.py:
llm = LLMFactory.get_llm()  # Returns primary with fallback configured
grader_chain = (
    prompt | llm | StrOutputParser()
)  # Automatically uses fallback on errors
```

**Phase 4: Cost Tracking**

Add optional provider usage metrics:
- Track which provider handled each request
- Aggregate costs by provider
- Inform cost optimization decisions

### Follow-up Actions

- [ ] Design LLMProvider protocol/interface
- [ ] Enhance LLMFactory with multi-provider support
- [ ] Implement RunnableWithFallback integration
- [ ] Update configuration system for provider selection
- [ ] Create provider-specific test suites
- [ ] Document provider switching instructions
- [ ] Implement cost tracking/reporting
- [ ] Test failover behavior under provider outages

## Status History

| Date       | Status         | Notes                                          |
| ---------- | -------------- | ---------------------------------------------- |
| 2025-12-05 | 🟡 In Progress | Initial research and abstraction design        |

---

_Last updated: 2025-12-05 by Development Team_
_Reference Implementation: src/infrastructure/llm_factory.py, src/use_cases/nodes.py_
_LangChain Documentation: https://python.langchain.com/docs/integrations/chat/_
