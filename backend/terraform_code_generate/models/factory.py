import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage

from backend.terraform_code_generate.config.config import AgentConfig,get_app_config

load_dotenv(encoding="utf-8")

def create_code_generate_model(config: AgentConfig) -> BaseChatOpenAI:
    common_params = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "streaming": True,
    }

    # OpenAI
    if config.model_type == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            **common_params
        )
    # xiaomi
    elif config.model_type == "xiaomi":
        return ChatOpenAI(
            model=os.getenv("XIAOMI_MODEL"),
            api_key=os.getenv("XIAOMI_API_KEY"),
            base_url=os.getenv("XIAOMI_OPENAI_BASE_URL"),
            # extra_body={"thinking": {"type": "enabled"}},
            **common_params
        )
    # Deepseek
    elif config.model_type == "deepseek":
        return ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            extra_body={
                "enable_thinking": True,
                "return_reasoning": True,
            },

            **common_params
        )
    # GLM
    elif config.model_type == "glm":
        return ChatOpenAI(
            model=os.getenv("GLM_MODEL"),
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL"),
            **common_params
        )
    # Qwen
    elif config.model_type == "qwen":
        return ChatOpenAI(
            model=os.getenv("QWEN_MODEL"),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
            **common_params
        )
    else:
        raise ValueError(f"不支持的模型类型：{config.model_type}")


def call_model(user_input: str, config: AgentConfig):
    """
    统一调用任意模型，返回回答内容
    :param config: 模型类型
    :param user_input: 用户输入
    :return: 模型回答
    """
    # 1. 初始化模型（统一接口）
    model = create_code_generate_model(config)

    # 2. 构造消息（统一格式）
    messages = [
        SystemMessage(content="你是一个友好的中文助手，回答简洁明了"),
        HumanMessage(content=user_input)
    ]

    # 3. 调用模型（统一方法）
    response = model.invoke(messages)

    # 4. 返回结果（统一输出）
    return response.content

if __name__ == "__main__":
    # 测试用例
    user_question = "什么是Python的装饰器？用简单的例子说明"

    # 测试不同模型（只需修改model_type参数）
    cfg = get_app_config()
    answer = call_model( user_question, cfg)
    print(f"回答：{answer}")

