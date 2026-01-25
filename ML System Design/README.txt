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
	