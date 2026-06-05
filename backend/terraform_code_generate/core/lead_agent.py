class PlanSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体

    这个Agent能够：
    1. 将复杂问题分解为简单步骤（使用 Function Calling）
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    5. 支持工具调用（可选）

    特别适合多步骤推理、数学问题、复杂分析等任务。
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            planner_prompt: Optional[str] = None,
            executor_prompt: Optional[str] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True,
            max_tool_iterations: int = 3
    ):
        """
        初始化PlanSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词（Agent级别）
            config: 配置对象
            planner_prompt: 规划器的系统提示词（可选）
            executor_prompt: 执行器的系统提示词（可选）
            tool_registry: 工具注册表（可选）
            enable_tool_calling: 是否启用工具调用
            max_tool_iterations: 最大工具调用迭代次数
        """
        # 传递 tool_registry 到基类
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            tool_registry=tool_registry
        )

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(
            self.llm,
            executor_prompt,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations
        )

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent

        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")

            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

            return final_answer

        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    async def arun_stream(
            self,
            input_text: str,
            on_start: LifecycleHook = None,
            on_finish: LifecycleHook = None,
            on_error: LifecycleHook = None,
            **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        PlanAgent 真正的流式执行

        实时返回：
        - 规划阶段的计划生成
        - 执行阶段的每个步骤输出

        Args:
            input_text: 用户输入
            on_start: 开始钩子
            on_finish: 完成钩子
            on_error: 错误钩子
            **kwargs: 其他参数

        Yields:
            StreamEvent: 流式事件
        """
        # 发送开始事件
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text
        )

        try:
            # 阶段 1：规划
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="planning",
                description="生成执行计划"
            )

            print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

            # 生成计划（同步方法，暂时保持）
            plan = self.planner.plan(input_text, **kwargs)

            if not plan:
                error_msg = "无法生成有效的行动计划，任务终止。"

                yield StreamEvent.create(
                    StreamEventType.ERROR,
                    self.name,
                    error=error_msg,
                    phase="planning"
                )

                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=error_msg
                )

                self.add_message(Message(input_text, "user"))
                self.add_message(Message(error_msg, "assistant"))
                return

            yield StreamEvent.create(
                StreamEventType.STEP_FINISH,
                self.name,
                phase="planning",
                plan=plan,
                total_steps=len(plan)
            )

            # 阶段 2：执行计划
            step_results = []

            for i, step_description in enumerate(plan):
                step_num = i + 1

                # 步骤开始
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="execution",
                    step=step_num,
                    total_steps=len(plan),
                    description=step_description
                )

                print(f"\n--- 步骤 {step_num}/{len(plan)} ---")
                print(f"📋 {step_description}")

                # 构建执行提示
                context = "\n".join([
                    f"步骤 {j + 1}: {plan[j]} -> {step_results[j]}"
                    for j in range(len(step_results))
                ])

                prompt = f"""原始问题: {input_text}

完整计划:
{chr(10).join([f"{j + 1}. {s}" for j, s in enumerate(plan)])}

已完成的步骤:
{context if context else "无"}

当前步骤: {step_description}

请执行当前步骤并给出结果。"""

                messages = [{"role": "user", "content": prompt}]

                # 流式执行步骤
                step_result = ""
                async for chunk in self.llm.astream_invoke(messages, **kwargs):
                    step_result += chunk

                    yield StreamEvent.create(
                        StreamEventType.LLM_CHUNK,
                        self.name,
                        chunk=chunk,
                        phase="execution",
                        step=step_num
                    )

                    print(chunk, end="", flush=True)

                print()  # 换行

                step_results.append(step_result)

                # 步骤完成
                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="execution",
                    step=step_num,
                    result=step_result
                )

            # 生成最终答案
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="final_answer",
                description="生成最终答案"
            )

            final_prompt = f"""原始问题: {input_text}

执行计划和结果:
{chr(10).join([f"{i + 1}. {plan[i]} -> {step_results[i]}" for i in range(len(plan))])}

请基于以上步骤的执行结果，给出原始问题的最终答案。"""

            final_messages = [{"role": "user", "content": final_prompt}]

            final_answer = ""
            async for chunk in self.llm.astream_invoke(final_messages, **kwargs):
                final_answer += chunk

                yield StreamEvent.create(
                    StreamEventType.LLM_CHUNK,
                    self.name,
                    chunk=chunk,
                    phase="final_answer"
                )

            # 发送完成事件
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=final_answer,
                total_steps=len(plan)
            )

            print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

            # 保存到历史
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

        except Exception as e:
            # 发送错误事件
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise