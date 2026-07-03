import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import Checkpointer

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.config.config import AgentConfig
from backend.terraform_code_generate.middlewares import  LoggingMiddleware, TokenUsageMiddleware, ContextSummarizationMiddleware
from backend.terraform_code_generate.plan_and_execute.fixed_step.prompt import PLANNER_PROMPT_TEMPLATE
from backend.terraform_code_generate.plan_and_execute.fixed_step.response import FixedStepPlannerResponse
from backend.terraform_code_generate.plan_and_execute.graph.build_graph import build_data_source_graph
from backend.terraform_code_generate.plan_and_execute.planner import Planner
from backend.terraform_code_generate.plan_and_execute.response import PlannerResponse

logger = logging.getLogger(__name__)

AGENT_NAME = "fixed_step_planner_agent"

class FixedStepPlanner(Planner):
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer,
            agent_config: AgentConfig,
    ):
        super(self.__class__, self).__init__(model, config, check_pointer, agent_config, AGENT_NAME)

    def plan(self, agent_state: CodeAgentState) -> FixedStepPlannerResponse | None:
        """
        生成执行计划
        """
        print(f"\n--- begin to generate plan ---")

        try:
            agent = super().create_planner_agent(PlannerResponse)

            result = agent.invoke(
                input={
                    "messages": [HumanMessage(content=agent_state["request_message"])]
                },
                config=self.config,
            )

            resource_type = result["structured_response"].resource_type
            # if resource_type == "data_source":
            #     graph = build_data_source_graph(
            #         agent_config=self.agent_config,
            #         model=self.model,
            #         config=self.config,
            #         check_pointer=self.check_pointer,
            #     )
            #     # return graph
            # elif resource_type == "resource":
            #     pass
            # else:
            #     print(f"\n ❌ resource type {resource_type} is not supported:")
            #     raise ValueError(f"not supported resource type：{resource_type}")


            print(f"\n ✅ generate plan complete ")

            steps = ["generate_code", "generate_test", "generate_doc"]
            return FixedStepPlannerResponse(resource_type=resource_type,steps=steps)

        except Exception as e:
            print(f"\n ❌ generate plan fail: {e}")
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
