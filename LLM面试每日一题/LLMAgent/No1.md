## # No.1

## Question:
### <mark>Agent Interview Question: How to design the overall architecture of an internal knowledge base Q&A Agent?</mark>

## Answer:

## 📋 Requirements

### Business Background:

The company has a set of Confluence documents and wants to build an "internal knowledge assistant" to support employees in asking questions like "What is the reimbursement process?" or "How to apply for a computer?"

### Design Requirements:

• Draw a complete Agent architecture diagram on the whiteboard: LLM core, planner (if needed), memory, tools (RAG/search), environment interface (HTTP/SDK)

• Explain the key responsibilities of each module in the code (including classes), and the data flow between modules

### Key Evaluation Points:

• Can you completely list the main components of a practical agent, rather than just "an LLM + vector database"

• Abstraction layers: API layer / Orchestrator layer / LLM client layer / Memory & Tools layer

• Understanding of "maintaining state" and "progressive cost invocation"


## 💡 Answer

### Core Architecture: ReAct + Plan-Execute Hybrid Mode ⚖️

## 1. Agentic Architecture Design

Modern knowledge base Agents are not simply "search + generation", but intelligent entities with **autonomous planning**, **reflection**, and **iterative optimization** capabilities.

### 💡 Answer Ideas:

During interviews, avoid two extremes: one is oversimplification (only "User → RAG → LLM → Response"), the other is over-design (stacking dozens of Agent modules).

**Recommended answer strategy shows progressive thinking:**

1. First explain the baseline approach (search + generation), acknowledging this works for simple scenarios
2. Then point out pain points (unstable search quality, unable to handle complex queries)
3. Then introduce ReAct mode, explaining how to solve these problems through planning and reflection
4. Finally propose extensibility (can evolve into Multi-Agent in the future)

This architecture may seem somewhat complex, but it's actually the **foundational mode** of Agentic design: Planner decomposes tasks, Executor performs specific operations, Reflector evaluates result quality. The three roles have clear division of labor and unified responsibilities, reflecting understanding of Agent mode without over-design. In interviews, this approach demonstrates your judgment of production environments—knowing when to use what architecture.


# ReAct Architecture Diagram

## System Flow:

```
User Query (用户查询)
    ↓
Routing Agent (路由Agent)
    ↓
Planning Agent (规划Agent) → Task Decomposition (任务分解)
    ↓
Execution Agent (执行Agent)
    ↓
    ├── RAG Search (RAG检索)
    ├── Internal API (内部API)
    └── Reflection Agent (反思Agent) → Requires Improvement (需要改进)
    ↓
Memory System (记忆系统)
    ↓
Output Answer (输出答案) → back to User Query
```

## 2. Core Concept: Thinking-Acting-Reflecting

### ReAct Loop:

**Thought → Action → Observation → Reflection**

• **Thought**: Analyze user intent, plan search strategy

• **Action**: Execute RAG search, invoke tools

• **Observation**: Evaluate search quality

• **Reflection**: Determine if re-search is needed
# Framework Selection and Implementation 🛠️

## 1. LangGraph: State-Driven Agent

### Why Choose LangGraph?

• Explicit state management, supports complex workflows

• Supports loops, conditional branching, and parallel execution

• Built-in checkpointing and error recovery

### Code Example:

````python
from langgraph.graph import StateGraph
from langchain.agents import create_react_agent

# Define state
class AgentState(TypedDict):
    messages: List[Message]
    plan: str
    retrieved_docs: List[Doc]

# Build graph
graph = StateGraph(AgentState)
graph.add_node("planner", plan_node)
graph.add_node("retriever", rag_node)
graph.add_node("reflect", reflect_node)
graph.add_edge("planner", "retriever")
graph.add_conditional_edges(
    "reflect",
    should_continue,  # Determine whether to continue
    {"continue": "retriever", "end": END}
)

## 2. Memory System: Short-term + Long-term Hybrid

### Dual-layer Memory Architecture:

| Type | Storage | Purpose | Scope |
|------|---------|---------|-------|
| Short-term | Redis | Conversation context | Current conversation |
| Long-term | Pgvector | User preferences/FAQ | All conversations |

### Memory Retrieval Strategy:

````python
# LangChain Memory integration
from langchain.memory import (
    ConversationBufferMemory,
    VectorStoreRetrieverMemory
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
long_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
)
