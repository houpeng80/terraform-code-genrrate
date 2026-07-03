from typing import Any, List

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.types import Checkpointer

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.agents.code_agent.data_source_agent.data_source_code_generate import \
    DataSourceCodeGenerate
from backend.terraform_code_generate.agents.docs_agents.data_source_agent.data_source_doc_generate import \
    DataSourceDocGenerate
from backend.terraform_code_generate.agents.test_agent.data_source_agent.data_source_test_generate import \
    DataSourceTestGenerate
from backend.terraform_code_generate.middlewares import LoggingMiddleware, TokenUsageMiddleware, ContextSummarizationMiddleware
from backend.terraform_code_generate.plan_and_execute.executor import Executor

AGENT_NAME = "fixed_step_executor_agent"

class FixedStepExecutor(Executor):
    """执行器 - 负责将执行规划期规划的步骤"""

    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer,
    ):
        super(self.__class__, self).__init__(model, config, check_pointer, AGENT_NAME)

    def execute(self, agent_state: CodeAgentState, steps: List[str]) -> CodeAgentState | None:
        """按计划执行任务"""
        print(f"\n--- begin to execute fixed plan ---")

        try:
            tasks = {
                "generate_code": DataSourceCodeGenerate(
                        self.model,
                        self.config,
                        self.check_pointer
                    ).generate,
                "generate_test": DataSourceTestGenerate(
                    self.model,
                    self.config,
                    self.check_pointer
                ).generate,
                "generate_doc": DataSourceDocGenerate(
                    self.model,
                    self.config,
                    self.check_pointer
                ).generate
            }

            # with ThreadPoolExecutor(max_workers=5) as executor:
            #     futures = [executor.submit(tasks[step], agent_state) for step in steps]
            #     done, not_done = wait(futures, return_when=ALL_COMPLETED)  # 等待所有任务完成
            #     for future in done:
            #         print(future.result())

            # with ThreadPoolExecutor(max_workers=5) as executor:
            #
            #     futures = [executor.submit(tasks[step]) for step in steps]
            #     for future in as_completed(futures):
            #         print(future.result())

            for step in steps:
                if step == "generate_code":
                    result_state = DataSourceCodeGenerate(
                        self.model,
                        self.config,
                        self.check_pointer
                    ).generate(agent_state)
                if step == "generate_test":
                    result_state = DataSourceTestGenerate(
                        self.model,
                        self.config,
                        self.check_pointer
                    ).generate(agent_state)
                if step == "generate_doc":
                    result_state = DataSourceDocGenerate(
                        self.model,
                        self.config,
                        self.check_pointer
                    ).generate(agent_state)



            print(f"\n ✅ execute plan complete ")
        except Exception as e:
            print(f"\n ❌ execute plan fail: {e}")
            return None

        return agent_state

    def build_middlewares(self) -> list[AgentMiddleware]:
        middlewares: list[AgentMiddleware] = [
            LoggingMiddleware(agent_name=AGENT_NAME),
            TokenUsageMiddleware(agent_name=AGENT_NAME),
            ContextSummarizationMiddleware(model=self.model, agent_name=AGENT_NAME),
        ]
        return middlewares
