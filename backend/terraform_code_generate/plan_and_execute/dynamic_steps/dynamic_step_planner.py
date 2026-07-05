import logging
from typing import List, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import Checkpointer

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.agents.code_agent.data_source_agent.data_source_code_generate import \
    DataSourceCodeGenerate
from backend.terraform_code_generate.agents.code_agent.resource_agent.resource_generate_code import ResourceCodeGenerate
from backend.terraform_code_generate.agents.docs_agents.data_source_agent.data_source_doc_generate import \
    DataSourceDocGenerate
from backend.terraform_code_generate.agents.docs_agents.resource_agent.resource_doc_generate import ResourceDocGenerate
from backend.terraform_code_generate.agents.generate import Generate
from backend.terraform_code_generate.agents.test_agent.data_source_agent.data_source_test_generate import \
    DataSourceTestGenerate
from backend.terraform_code_generate.agents.test_agent.resource_agent.resource_test_generate import ResourceTestGenerate
from backend.terraform_code_generate.config.config import AgentConfig
from backend.terraform_code_generate.middlewares import  LoggingMiddleware, TokenUsageMiddleware, ContextSummarizationMiddleware
from backend.terraform_code_generate.plan_and_execute.dynamic_steps.prompt import PLANNER_PROMPT_TEMPLATE
from backend.terraform_code_generate.plan_and_execute.dynamic_steps.response import DynamicStepPlannerResponse
from backend.terraform_code_generate.plan_and_execute.planner import Planner
from client.generate_test import generate_code

logger = logging.getLogger(__name__)

AGENT_NAME = "dynamic_step_planner_agent"


class ReourceDocGenerate:
    pass


class DynamicStepPlanner(Planner):
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer,
            agent_config: AgentConfig,
    ):
        super(self.__class__, self).__init__(model, config, check_pointer, agent_config, AGENT_NAME)

    def plan(self, agent_state: CodeAgentState) -> DynamicStepPlannerResponse | None:
        """
        生成执行计划
        """
        print(f"\n--- begin to generate plan ---")

        try:
            agent = super().create_planner_agent(DynamicStepPlannerResponse)

            result = agent.invoke(
                input={
                    "messages": [HumanMessage(content=agent_state["request_message"])]
                },
                config=self.config,
            )

            resource_type = result["structured_response"].resource_type

            steps = []
            if resource_type == "data_source":
                for step in result["structured_response"].steps:
                    if step == "generate_code":
                        if not self.agent_config.generate_code:
                            print(f"\n ❌ {step} is not supported, please change the parameter `generate_code` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(DataSourceCodeGenerate(self.model, self.config, self.check_pointer))
                    elif step == "generate_test":
                        if not self.agent_config.generate_test:
                            print(f"\n ❌ {step} is not supported, please change the parameter `generate_test` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(DataSourceTestGenerate(self.model, self.config, self.check_pointer))
                    elif step == "generate_doc":
                        if not self.agent_config.generate_doc:
                            print(f"\n ❌ {step} is not supported, please change the parameter `generate_doc` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(DataSourceDocGenerate(self.model, self.config, self.check_pointer))
                    else:
                        print(f"\n ❌ {step} is not supported:")
                        raise ValueError(f"not supported step：{step}")
            elif resource_type == "resource":
                for step in result["structured_response"].steps:
                    if step == "generate_code":
                        if not self.agent_config.generate_code:
                            print(
                                f"\n ❌ {step} is not supported, please change the parameter `generate_code` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(ResourceCodeGenerate(self.model, self.config, self.check_pointer))
                    elif step == "generate_test":
                        if not self.agent_config.generate_test:
                            print(
                                f"\n ❌ {step} is not supported, please change the parameter `generate_test` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(ResourceTestGenerate(self.model, self.config, self.check_pointer))
                    elif step == "generate_doc":
                        if not self.agent_config.generate_doc:
                            print(
                                f"\n ❌ {step} is not supported, please change the parameter `generate_doc` value to `true` in the file `config.yaml`")
                            raise ValueError(f"not supported step：{step}")
                        steps.append(ResourceDocGenerate(self.model, self.config, self.check_pointer))
                    else:
                        print(f"\n ❌ {step} is not supported:")
                        raise ValueError(f"not supported step：{step}")
            else:
                print(f"\n ❌ resource type {resource_type} is not supported:")
                raise ValueError(f"not supported resource type：{resource_type}")

            print(f"\n ✅ generate dynamic plan complete ")

            return DynamicStepPlannerResponse(resource_type=resource_type, steps=steps)

        except Exception as e:
            print(f"\n ❌ generate dynamic plan fail: {e}")
            return None

    def build_system_prompt_template(self) -> str:
        return PLANNER_PROMPT_TEMPLATE.format(agent_name = AGENT_NAME)

    def build_middlewares(self) -> list[AgentMiddleware]:
        middlewares: list[AgentMiddleware] = [
            LoggingMiddleware(agent_name=AGENT_NAME),
            TokenUsageMiddleware(agent_name=AGENT_NAME),
            ContextSummarizationMiddleware(model=self.model, agent_name=AGENT_NAME),
        ]
        return middlewares
