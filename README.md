# TerraformCodeGenerateAgent

TerraformCodeGenerateAgent 是一个基于 OpenAI/Anthropic 原生 API 构建的生产级多智能体框架，集成了工具响应协议（ToolResponse）、上下文工程（HistoryManager/TokenCounter）、会话持久化（SessionStore）、子代理机制（TaskTool）、乐观锁（文件编辑）、熔断器（CircuitBreaker）、Skills 知识外化、TodoWrite 进度管理、DevLog 决策记录、流式输出（SSE）、异步生命周期、可观测性（TraceLogger）、日志系统（四种范式）、LLM/Agent 基类重构等 16 项核心能力，为构建复杂智能体应用提供完整的工程化支持。
