import logging

from typing import Any
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from backend.terraform_code_generate.agents.extract_web_info_agent.prompt import apply_prompt_template
from backend.terraform_code_generate.config.config import get_app_config
from backend.terraform_code_generate.middlewares import TokenUsageMiddleware
from backend.terraform_code_generate.models import create_code_generate_model
from backend.terraform_code_generate.tools.web_search import web_search

logger = logging.getLogger(__name__)

AGENT_NAME = "web_search_and_extract_agent"

class WebSearchAndExtractInfo(BaseModel):
    is_global: bool = Field(description="Whether the service is global service.")
    service_name: str = Field(description="The service name")
    uri: str = Field(description="The URI address of the API")
    uri_params: Any = Field(description="The URI params of the API")
    query_params: Any = Field(description="The query params of the API")
    request_params: Any = Field(description="The request params of the API")
    response_params: Any = Field(description="The response params of the API")
    page_info: Any = Field(description="The page info of the API")


class ApiInfo(BaseModel):
    api_info: WebSearchAndExtractInfo = Field(default_factory=WebSearchAndExtractInfo, description="The API info extracted from the API")

class WebSearchAndExtract:
    """get web info and extra information."""

    def __init__(self):
        self.agent_name = AGENT_NAME

    def build_system_prompt_template(self) -> str:
        return apply_prompt_template(self.agent_name)

    def web_search_and_extract(self, request: str) -> Any:
        """
        从web中获取网页信息，并从中提取关键信息
        Args:
            request: 用户的需求

        Returns:
            提取到的API信息
        """
        logger.info("agent {%s} begin to get API info", AGENT_NAME)

        config = get_app_config()
        model = create_code_generate_model(config)
        user_message = HumanMessage(
            content=request
        )
        agent = create_agent(
            name=self.agent_name,
            model=model,
            system_prompt=self.build_system_prompt_template(),
            tools=[web_search],
            middleware=[TokenUsageMiddleware(agent_name=self.agent_name)],
            response_format=ApiInfo,
        )

        try:
            result = agent.invoke(
                {"messages": [user_message]},
            )

            logger.info("agent {%s} get the API info complete", AGENT_NAME)

            return result["structured_response"].api_info
        except Exception as e:
            print(f"\n ❌ agent {AGENT_NAME} get the API info fail: {e}")
            return None
