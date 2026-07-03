from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ResponseT
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph.state import StateGraph
from langgraph.types import Checkpointer, StateT, OutputT
from langgraph.typing import ContextT, InputT

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.config.config import AgentConfig

class Replanner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer,
            agent_config: AgentConfig,
            agent_name: str
    ):
        self.model = model
        self.agent_name = agent_name
        self.config = config
        self.check_pointer = check_pointer
        self.agent_config = agent_config

    def replan(self, agent_state: CodeAgentState) ->  StateGraph[StateT, ContextT, InputT, OutputT] | None:
        pass

    def build_middlewares(self, agent_state: CodeAgentState) -> list[AgentMiddleware]:
        pass

    def build_system_prompt_template(self) -> str:
        pass