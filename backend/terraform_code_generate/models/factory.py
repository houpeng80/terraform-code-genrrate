import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(encoding="utf-8")

def create_code_generate_model(model_type: str):

    common_params = {
        "temperature": 0,
        "max_tokens": 1024,
        "timeout": 30
    }

    # OpenAI
    if model_type == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            **common_params
        )

    # Anthropic Claude
    elif model_type == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("XIAOMI_MODEL"),
            api_key=os.getenv("XIAOMI_API_KEY"),
            base_url=os.getenv("XIAOMI_ANTHROPIC_BASE_URL"),
            **common_params
        )

    # Deepseek
    elif model_type == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            **common_params
        )

    # Qwen
    elif model_type == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("QWEN_MODEL"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            **common_params
        )
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")


def call_model(model_type: str, user_input: str):
    """
    统一调用任意模型，返回回答内容
    :param model_type: 模型类型
    :param user_input: 用户输入
    :return: 模型回答
    """
    # 1. 初始化模型（统一接口）
    model = create_code_generate_model(model_type)

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
    for model_type in ["openai", "claude", "deepseek" ]:
        try:
            print(f"\n===== 测试 {model_type} 模型 =====")
            answer = call_model(model_type, user_question)
            print(f"回答：{answer}")
        except Exception as e:
            print(f"调用失败：{str(e)}")

