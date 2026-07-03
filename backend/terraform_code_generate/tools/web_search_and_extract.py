
from langchain.tools import tool

from backend.terraform_code_generate.agents.extract_web_info_agent.extract_web_info_agent import WebSearchAndExtract


@tool
def web_search_and_extract(url: str):
    """从指定API地址查询所需要的数据，并且从获取到的结果中提取API的信息。
    当需要根据API地址获取网页信息时触发

    Args:
        url: 要查询的地址
    """

    result = WebSearchAndExtract().web_search_and_extract(url)
    return result

if __name__ == "__main__":
    api = """
    API：https://support.huaweicloud.com/api-gaussdb/gaussdb_api_219.html
    """

    test = WebSearchAndExtract()
    res = test.web_search_and_extract(api)
    print(res)
