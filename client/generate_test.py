import uuid

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

from backend.terraform_code_generate.agents.code_agent.data_source_agent.data_source_code_generate import \
    DataSourceCodeGenerate
from backend.terraform_code_generate.agents.docs_agents.data_source_agent.data_source_doc_generate import \
    DataSourceDocGenerate
from backend.terraform_code_generate.agents.test_agent.data_source_agent.data_source_test_generate import \
    DataSourceTestGenerate
from backend.terraform_code_generate.config.config import get_app_config
from backend.terraform_code_generate.models import create_code_generate_model
from backend.terraform_code_generate.plan_and_execute.generate_leader import GenerateLeader
from backend.terraform_code_generate.plan_and_execute.graph.graph_executor import GraphExecutor
from backend.terraform_code_generate.plan_and_execute.graph.graph_planner import GraphPlanner


# def code_check(code: str):
#     app_config = get_app_config()
#     test_model = create_code_generate_model(app_config)
#     runnable_config = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_id": str(uuid.uuid4())}}
#     checkpointer = InMemorySaver()
#     agentState = {
#         "code_result": code,
#     }
#
#     code_check = DataSourceCodeCheck(test_model, runnable_config, checkpointer)
#     result = code_check.code_check(agentState)
#     print(result)

def generate_code(message: str):
    app_config = get_app_config()
    test_model = create_code_generate_model(app_config)
    runnable_config = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_id": str(uuid.uuid4())}}
    checkpointer = InMemorySaver()
    agentState = {
        "request_message": message,
        "code_retries_time": 0,
        "test_retries_time": 0,
        "doc_retries_time": 0,
        "current_step": "generate_code",
        "resource_type": "data_source"
    }

    code_generator = DataSourceCodeGenerate(test_model, runnable_config, checkpointer)
    code_generator.generate(agentState)

def generate_test(message: str):
    app_config = get_app_config()
    test_model = create_code_generate_model(app_config)
    runnable_config = {"configurable": {"thread_id": "my_test_check_pointer"}}
    checkpointer = InMemorySaver()
    agentState = {
        "request_message": message,
        "code_retries_time": 0,
        "test_retries_time": 0,
        "doc_retries_time": 0,
        "current_step": "generate_test",
        "resource_type": "data_source"
    }

    code_generator = DataSourceTestGenerate(test_model, runnable_config, checkpointer)
    code_generator.generate(agentState)

def generate_doc(message: str):
    app_config = get_app_config()
    test_model = create_code_generate_model(app_config)
    runnable_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    checkpointer = InMemorySaver()
    agentState = {
        "request_message": message,
        "code_retries_time": 0,
        "test_retries_time": 0,
        "doc_retries_time": 0,
        "input_token_statistics": 0,
        "output_token_statistics": 0,
        "total_token_statistics": 0,
        "resource_type": "data_source"
    }

    code_generator = DataSourceDocGenerate(test_model, runnable_config, checkpointer)
    code_generator.generate(agentState)

def planner_plan(message: str):
    app_config = get_app_config()
    test_model = create_code_generate_model(app_config)
    runnable_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    checkpointer = InMemorySaver()
    initial_state = {
        "request_message": message,
        "code_retries_time": 0,
        "test_retries_time": 0,
        "doc_retries_time": 0,
        "input_token_statistics": 0,
        "output_token_statistics": 0,
        "total_token_statistics": 0,
        # "resource_type": "data_source"
    }

    graph_planner = GraphPlanner(test_model, runnable_config, checkpointer, app_config)
    graph = graph_planner.plan(initial_state)
    # graph.invoke(initial_state, runnable_config)

    print("=============================================")
    # png_data = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)


def planner_execute(message: str):
    app_config = get_app_config()
    test_model = create_code_generate_model(app_config)
    runnable_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    checkpointer = InMemorySaver()
    initial_state = {
        "request_message": message,
        "code_retries_time": 0,
        "test_retries_time": 0,
        "doc_retries_time": 0,
        # "resource_type": "data_source"
    }

    graph_planner = GraphPlanner(test_model, runnable_config, checkpointer, app_config)
    plan_res = graph_planner.plan(initial_state)

    initial_state["resource_type"] = plan_res["resource_type"]

    graph_executor = GraphExecutor(test_model, runnable_config, checkpointer)
    graph_executor.execute(initial_state, plan_res["graph"])

    print("=============================================")
    png_data = plan_res["graph"].compile().get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

def generate(message: str):
    leader = GenerateLeader()
    leader.run(message)

if __name__ == "__main__":
    data_source_gaussdb = """
    根据API：https://support.huaweicloud.com/intl/zh-cn/api-gaussdb/gaussdb_api_524.html，帮我生成一个data source
    """

    data_source_gaussdb_post = """
        根据API：https://support.huaweicloud.com/api-gaussdb/gaussdb_api_314.html，帮我生成一个data source
        """

    data_source_dcs = """
    根据API：https://support.huaweicloud.com/api-dcs/ListBigkeyScanTasks.html，帮我生成一个data source
    """


    apis = [

        # rds
        "https://support.huaweicloud.com/api-rds/rds_01_0004.html",
        "https://support.huaweicloud.com/api-rds/rds_12_0014.html",
        "https://support.huaweicloud.com/api-rds/rds_12_0016.html",
        "https://support.huaweicloud.com/api-rds/rds_12_0010.html",
        "https://support.huaweicloud.com/api-rds/rds_09_0017.html",
        "https://support.huaweicloud.com/api-rds/rds_08_0039.html",
        "https://support.huaweicloud.com/api-rds/rds_06_0056.html",
        "https://support.huaweicloud.com/api-rds/rds_06_0077.html",

        # dcs
        "https://support.huaweicloud.com/api-dcs/ListInstances.html",
        "https://support.huaweicloud.com/api-dcs/ShowNodesInformation.html",
        "https://support.huaweicloud.com/api-dcs/ListMigrationTask.html",
        "https://support.huaweicloud.com/api-dcs/ListBackgroundTask.html",
        "https://support.huaweicloud.com/api-dcs/ListRedislog.html",

        # dds
        "https://support.huaweicloud.com/api-dds/dds_connect_0002.html",
        "https://support.huaweicloud.com/api-dds/dds_api_0266.html",

        # elb
        "https://support.huaweicloud.com/api-elb/ListListeners.html",
        "https://support.huaweicloud.com/api-elb/ListLoadBalancers.html",

        # vpn
        "https://support.huaweicloud.com/api-vpn/vpn_api_0023.html",

        # gaussdb
        "https://support.huaweicloud.com/api-gaussdb/gaussdb_api_301.html",
        "https://support.huaweicloud.com/api-gaussdb/gaussdb_api_107.html",
        "https://support.huaweicloud.com/api-gaussdb/gaussdb_api_314.html",

        # geminidb
        "https://support.huaweicloud.com/api-nosql/nosql_05_0054.html",
        "https://support.huaweicloud.com/api-nosql/nosql_05_0068.html",

        # taurusdb
        "https://support.huaweicloud.com/api-taurusdb/ListParamsTemplateApplyHistory.html",
        "https://support.huaweicloud.com/api-taurusdb/ShowSlowLogStatistics.html",
        "https://support.huaweicloud.com/api-taurusdb/ListAuditLogs.html",

        # kafka
        "https://support.huaweicloud.com/api-kafka/ShowInstanceUsers.html",
        "https://support.huaweicloud.com/api-kafka/ListTopicProducers.html",
        "https://support.huaweicloud.com/api-kafka/ListInstances.html",
        "https://support.huaweicloud.com/api-kafka/ListConnectorTasks.html",
    ]
    # generate_code(data_source_message)
    # print("============================================")
    # generate_test(data_source_message)
    # print("============================================")
    # generate_doc(data_source_gaussdb)
    # print("============================================")
    # planner_plan(data_source_gaussdb)
    # print("============================================")
    # planner_execute(data_source_message)
    # print("============================================")
    # generate(data_source_gaussdb_post)

    for api in apis:
        request = f"根据API：{api}，帮我生成一个data source"
        generate(request)




    # client = OpenAI(
    #     api_key="sk-cd5d1d45908d42f2a29195e4671d7e1d",
    #     base_url="https://api.deepseek.com"
    # )
    #
    # response = client.chat.completions.create(
    #     model="deepseek-chat",  # 或 mimo-v2-flash
    #     messages=[{"role": "user", "content": "请计算斐波那契数列第10项"}],
    #     stream=True,
    #     extra_body={"thinking": {"type": "enabled"}}  # 关键：启用思考模式
    # )
    #
    # init_chat_model()
    #
    # print("=== 思考过程 ===")
    # for chunk in response:
    #     print(chunk)
    #     # delta = chunk.choices[0].delta
    #     # # 打印思考内容
    #     # if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
    #     #     print(delta.reasoning_content, end="", flush=True)
    #
    # print("\n=== 最终回答 ===")
