https://aiinterviewprep.substack.com/

# RAG
如何准备 才能过大厂RAG面试？
RAG是目前AI面试主题之一, 面试官从基础 → 进阶 → 高级方向层层递进考察：
1.  什么是RAG？它解决了什么问题？
RAG核心是通过外部知识库（向量数据库+嵌入模型）检索相关上下文，注入到LLM的prompt中生成答案。解决三大痛点：
•  hallucination on recent events
•  私有/企业内部知识无法直接训练
•  grounding with real docs
2.  RAG的完整pipeline长什么样？Indexing 文档加载 → 切块（chunking） → Embedding → 存入Vector DBQuery-time（在线）：用户query → Embed → 检索top-k → Rerank（可选） → 上下文拼接prompt → LLM生成 → 后处理
3.  RAG和Fine-tuning的区别
•  RAG：动态、实时更新知识、无需重训、成本低、可解释性强（可引用来源）、适合QA、客服、企业搜索
•  Fine-tuning：改变模型行为/风格/格式、学习新任务模式、适合领域垂直任务
4.  RAG效果不好 怎么优化？
•  Retrieval召回差 → 优化chunk策略（semantic chunk）、hybrid search（dense+sparse）、better embedding、metadata过滤
•  上下文噪声多 → Rerank（bge-reranker、Cohere Rerank）、上下文压缩
•  LLM幻觉 → Self-RAG/CRAG
•  长文档问题 → Parent-document retriever、small-to-big retrieval
•  评价指标
5.  Chunking策略有哪些？
•  Fixed size
•  Recursive/Sentence/Paragraph
•  Semantic chunking
•  Agentic chunking
6.  Reranking的作用？有哪些好用的reranker


# GenAI System Design 
我们上一期毕业的有学员回来问- 已收到硅谷大厂GenAI系统设计的面试邀约，如何准备？
备战GenAI系设面试时，关键在于展示如何将不确定的模型输出转化为稳定、可扩展的企业级服务
1. 掌握核心架构的“五层模型”
数据采集与处理层： 讨论如何处理非结构化数据。重点在于 Chunking（分块）策略（如固定长度、语义分块）、Embedding 模型的选择，以及数据清洗（去除噪声和重复）
检索与存储层： 讨论向量数据库（如 Pinecone, Milvus）的选型。强调混合搜索 (Hybrid Search)——即结合关键字（BM25）和语义向量，以及使用 Reranker（重排序） 来提升检索质量
业务编排层 (Orchestration)： 展现你对 CrewAI 或 LangChain 的理解。说明如何将复杂任务拆解为子任务，如何设计 Agent 的 ReAct (Reason + Act) 循环，以及如何处理状态管理。
模型服务层： 讨论模型的部署与权衡。什么时候使用托管 API（如 OpenAI），什么时候需要自建 vLLM 推理框架？讨论量化带来的速度提升与精度损耗
监控与护栏层： 讨论 LLM-as-a-Judge 的评估机制。如何防止幻觉？如何建立安全护栏（Guardrails）以过滤敏感信息或有害输出
2. 深度掌握关键技术点
面试官会期望你探讨
上下文管理： 如何处理长文本？讨论 KV Caching 技术如何加速推理，以及当上下文过长时如何通过“滑动窗口”或“摘要压缩”来节省 Token
成本与延迟： 这是一个系统架构师的必修课。你需要能够估算每秒查询数 (QPS) 所需的 GPU 数量，并讨论 TTFT (首字延迟) 对用户体验的影响
微调 vs RAG： 清楚地界定两者的边界。RAG 用于提供外部事实和实时数据；LoRA/QLoRA 微调用于改变模型语气、学习特定格式或掌握特定领域的专业术语
Agentic Workflows： 讨论从单次调用向“循环代理”的转变
3. 将项目经验转化为面试素材
隐私与安全
可靠性
实战痛点
4. 练习快速估算
显存占用
推理成本

# Blog to read 
https://eugeneyan.com/writing/llm-patterns/

https://lilianweng.github.io/posts/2023-06-23-agent/

https://huyenchip.com/2023/04/11/llm-engineering.html

https://cookbook.openai.com/

https://huyenchip.com/mlops/

https://www.youtube.com/watch?v=2ryjXysW6_c&list=PLLm69KFEX6JD4fJ7ge8-WEDw3G4c_nzHs&index=3

https://www.systemdesignhandbook.com/guides/ai-system-design/

https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#computer-interview-prep

Medium系列
https://medium.com/@vinodkrane/a-simple-6-step-framework-to-design-genai-systems-meet-scaled-9a5a34bee2a7#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6Ijk1NDRkMGZmMDU5MGYwMjUzMDE2NDNmMzI3NWJmNjg3NzY3NjU4MjIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDQ0MzI2MjQ1NzU2MTU3NTk3ODUiLCJlbWFpbCI6InhpYW5nenVvMjAxMkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibm9uY2UiOiJub3RfcHJvdmlkZWQiLCJuYmYiOjE3Njk0NzU0MzgsIm5hbWUiOiJFbWlseSBadW8iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSWRKbjFVcmVqZDM5T2VKY1JEbm1wcTFsZjl1bU5jS3hTbFhZX1hrWEJWb29CYTZDdHM9czk2LWMiLCJnaXZlbl9uYW1lIjoiRW1pbHkiLCJmYW1pbHlfbmFtZSI6Ilp1byIsImlhdCI6MTc2OTQ3NTczOCwiZXhwIjoxNzY5NDc5MzM4LCJqdGkiOiI1MDZkMmUxOWNhYjYyNGU4NWU4YjczZDYxMmY1YTIzN2Q2Y2RjZTQxIn0.k6ZZGvKi_FoDht3qmXqyuuH5GhkHL093xMYOeZEyooJC4-cmbHeA3z74IasNRHHXl0ZdZOEhMwP_EhvNhvJhKquDec8GNX6NEEDThJt2DnznqjASnXIi05rf2NO88bYLpqwKE-cMM8bNNpabTDOl85hIxD3AEiTj33ne8d3_ptHByMNHxgI14NxebxX2pwu2_16pO83ddinQWtg6yMfG7WtaqaXO5K2FLbJabcth4my3hcC20d44JjqqvF2_3TXhbNQyPu495lWz2ybGIQZvJH65fsSNgzakt8FZuitCGrpCkI3NN0POM7buL5fVbtOtO22YYhX5mWnD9WIf6uDUTw

https://www.youtube.com/watch?v=mEsleV16qdo

https://jalammar.github.io/illustrated-transformer/

GenAI系统设计核心话题
面试中常见的设计问题：
RAG系统:

Chunking策略和embedding选择
Hybrid search (vector + keyword)
Reranking和context compression
评估检索质量

Agent架构:

ReAct, Plan-and-Execute等模式
Tool/function calling设计
Memory管理(短期/长期)
Multi-agent coordination

LLM Serving:

Batching和KV cache优化
Model parallelism (tensor/pipeline)
Quantization和pruning
Cost vs latency权衡

Prompt Engineering系统化:

Few-shot learning策略
Chain-of-thought prompting
Prompt versioning和A/B测试
Guardrails和safety layers

评估和监控:

LLM-as-judge评估框架
Hallucination detection
Cost tracking和token优化
User feedback循环

实战案例和博客:

OpenAI Cookbook - 官方最佳实践
Anthropic's Claude documentation - Prompt engineering和系统集成
"Building LLM applications for production" (Hazy Research)
a16z的AI工程博客 - 行业视角和架构模式


Google DeepMind 
ML深度 - 你需要补的重点
会被问到的内容
基础理论（必须滚瓜烂熟）

Transformer详细原理

Self-attention数学推导
Multi-head attention为什么有效
Positional encoding的作用
Layer norm vs Batch norm


优化算法

Adam, AdamW区别
Learning rate scheduling
Gradient clipping


训练技巧

Dropout, regularization
Fine-tuning vs Pre-training
Transfer learning

深度问题（DeepMind特色）

"从零推导attention公式"
"解释BERT和GPT的区别及各自优势"
"如何训练一个100B参数的模型？"
"Distributed training的策略有哪些？"
"如何评估LLM的quality？"

你的项目相关

RAG系统的技术细节

Vector embedding选择
Retrieval策略
Re-ranking方法


Multi-agent系统架构
Vector database的实现原理

准备程度

必须读的论文（精读，能画图讲解）：

Attention is All You Need ⭐⭐⭐
BERT ⭐⭐⭐
GPT-2/3 ⭐⭐
RAG (Lewis et al.) ⭐⭐
InstructGPT/RLHF ⭐⭐


推荐读的论文（了解思路）：

Chinchilla, LLaMA, Gemini
Constitutional AI (Anthropic)
AutoGPT, MetaGPT

PyTorch实践

能从零实现：

Multi-head attention ⭐⭐⭐
简单的transformer encoder
Fine-tune一个BERT模型


了解：

PyTorch分布式训练
Mixed precision training
Model parallelism

Antropic 
System Design (1轮，45-60分钟)
设计分布式搜索系统：处理10亿文档、100万QPS，管理LLM推理超过1万次/秒 Linkjob
考察点

分片、缓存、LLM推理扩展、避免热点、GPU内存优化 Medium
可能问：设计让GPT在单个thread处理多个问题的系统 Interviewing
与Anthropic实际遇到的问题相关
准备重点

分布式搜索/推荐系统
LLM inference scaling
GPU memory优化
Vector database架构
⭐ AI Safety / Culture Fit
准备内容

读Constitutional AI论文
了解RLHF、Red teaming
准备2-3个safety-first的项目故事

核心问题

"如何在项目中平衡性能和安全？"
"AI可能带来的风险是什么？"
"为什么想加入Anthropic？"

Github for AI 2026 interviews
https://github.com/llmgenai/LLMInterviewQuestions
https://github.com/Devinterview-io?tab=repositories
https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub/blob/main/Interview_QA/QA_1-3.md
https://mimansajaiswal.github.io/posts/llm-ml-job-interviews-resources/
