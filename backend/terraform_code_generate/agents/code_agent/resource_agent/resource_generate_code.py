from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.types import Checkpointer

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.agents.generate import Generate
from backend.terraform_code_generate.config.config import get_app_config
from backend.terraform_code_generate.middlewares import LoggingMiddleware, TokenUsageMiddleware
from backend.terraform_code_generate.models import create_code_generate_model
from backend.terraform_code_generate.tools.skill_load import skill_load
from backend.terraform_code_generate.tools.web_search_and_extract import web_search_and_extract

AGENT_NAME = "resource_code_generator"

class ResourceCodeGenerate(Generate):
    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer
    ):
        super(self.__class__, self).__init__(model, config, check_pointer, AGENT_NAME)
        self.generate_type = "doc"
        self.current_step = "generating_doc"

    def build_system_prompt_template(self) -> str:
        return ""

    def build_middlewares(self) -> list[AgentMiddleware]:
        middlewares: list[AgentMiddleware] = [
            LoggingMiddleware(agent_name=AGENT_NAME),
            TokenUsageMiddleware(agent_name=AGENT_NAME),
            SummarizationMiddleware(model=self.model, keep=("messages", 6),trigger=[("tokens", 1000)]),
        ]
        return middlewares

    def build_tools(self) -> list[BaseTool]:
        return [web_search_and_extract, skill_load]

    def get_generate_type(self) -> str:
        return self.generate_type

    def get_current_step(self) -> str:
        return self.current_step
