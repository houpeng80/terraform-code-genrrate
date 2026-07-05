from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.types import Checkpointer

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.agents.generate import Generate
from backend.terraform_code_generate.agents.test_agent.resource_agent.prompt import apply_prompt_template
from backend.terraform_code_generate.middlewares import RetryCheckMiddleware
from backend.terraform_code_generate.middlewares.log_middleware import LoggingMiddleware
from backend.terraform_code_generate.middlewares.test_check_middleware import TestCheckMiddleware
from backend.terraform_code_generate.middlewares.token_usage_middleware import TokenUsageMiddleware
from backend.terraform_code_generate.middlewares.summarization_middleware import ContextSummarizationMiddleware
from backend.terraform_code_generate.middlewares.tool_cache_middleware import ToolCacheMiddleware
from backend.terraform_code_generate.tools.deal_file import write_file
from backend.terraform_code_generate.tools.web_search_and_extract import web_search_and_extract

AGENT_NAME = "data_source_test_generator"

class ResourceTestGenerate(Generate):
    def __init__(
            self,
            model: BaseChatOpenAI,
            config: RunnableConfig,
            check_pointer: Checkpointer
    ):
        super(self.__class__, self).__init__(model, config, check_pointer, AGENT_NAME)
        self.generate_type = "test"
        self.current_step = "generating_test"

    def build_system_prompt_template(self) -> str:
        # 项目根目录，用于放生成的文件
        repo_root = Path(__file__).resolve().parents[5]
        return apply_prompt_template(AGENT_NAME, repo_root.resolve())

    def build_middlewares(self) -> list[AgentMiddleware]:
        middlewares: list[AgentMiddleware] = [
            RetryCheckMiddleware(agent_name=AGENT_NAME, agent_config=self.agent_config),
            LoggingMiddleware(agent_name=AGENT_NAME),
            TokenUsageMiddleware(agent_name=AGENT_NAME),
            ToolCacheMiddleware(agent_name=AGENT_NAME),
            ContextSummarizationMiddleware(model=self.model, agent_name=AGENT_NAME),
            TestCheckMiddleware(agent_name=AGENT_NAME, agent_config=self.agent_config),
        ]
        return middlewares

    def build_tools(self) -> list[BaseTool]:
        return [web_search_and_extract, write_file]

    def get_generate_type(self) -> str:
        return self.generate_type

    def get_current_step(self) -> str:
        return self.current_step

