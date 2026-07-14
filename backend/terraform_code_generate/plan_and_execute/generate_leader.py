from typing import Any, List

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import StateT, OutputT
from langgraph.typing import ContextT, InputT

from backend.terraform_code_generate.agents.agent_state import CodeAgentState
from backend.terraform_code_generate.agents.generate import Generate
from backend.terraform_code_generate.config.config import get_app_config
from backend.terraform_code_generate.models import create_code_generate_model
from backend.terraform_code_generate.plan_and_execute.dynamic_steps.dynamic_step_executor import DynamicStepExecutor
from backend.terraform_code_generate.plan_and_execute.dynamic_steps.dynamic_step_planner import DynamicStepPlanner
from backend.terraform_code_generate.plan_and_execute.dynamic_steps.response import DynamicStepPlannerResponse
from backend.terraform_code_generate.plan_and_execute.fixed_step.fixed_step_executor import FixedStepExecutor
from backend.terraform_code_generate.plan_and_execute.fixed_step.fixed_step_planner import FixedStepPlanner
from backend.terraform_code_generate.plan_and_execute.fixed_step.response import FixedStepPlannerResponse
from backend.terraform_code_generate.plan_and_execute.graph.graph_executor import GraphExecutor
from backend.terraform_code_generate.plan_and_execute.graph.graph_planner import GraphPlanner
from backend.terraform_code_generate.plan_and_execute.graph.response import GraphPlannerResponse

class GenerateLeader:
    def __init__(self):
        agent_config = get_app_config()
        model = create_code_generate_model(agent_config)
        self.model = model
        self.agent_config = agent_config
        self.config = {"configurable": {"thread_id": "default_thread_id", "user_id": "default_user_id"}}
        self.check_pointer = InMemorySaver()

        if self.agent_config.execute_type == "graph":
            self.planner = GraphPlanner(model, self.config, self.check_pointer, agent_config)
            self.executor = GraphExecutor(model, self.config, self.check_pointer)
        elif self.agent_config.execute_type == "fixed_step":
            self.planner = FixedStepPlanner(model, self.config, self.check_pointer, agent_config)
            self.executor = FixedStepExecutor(model, self.config, self.check_pointer)
        elif self.agent_config.execute_type == "dynamic_step":
            self.planner = DynamicStepPlanner(model, self.config, self.check_pointer, agent_config)
            self.executor = DynamicStepExecutor(model, self.config, self.check_pointer)
        else:
            raise ValueError(f"\n--- 任务终止： 执行任务的类型{self.agent_config.execute_type}不正确，请修改后重试 --- ")

        self.agent_state = dict[str, Any]({
            "code_retries_time": 0,
            "test_retries_time": 0,
            "doc_retries_time": 0,
            "input_token_statistics": 0,
            "output_token_statistics": 0,
            "total_token_statistics": 0,
        })

    def run(self, question: str):
        print(f"\n--- begin to deal question: {question} ---")

        self.agent_state["request_message"] = question
        plan_response = self.planner.plan(self.agent_state)
        if not plan_response:
            print(f"\n--- ❌ task end, can not generate an effective action plan ---")
            return

        if isinstance(plan_response, GraphPlannerResponse):
            self.agent_state = self.execute_with_graph(self.agent_state, plan_response.resource_type, plan_response.graph)
        elif isinstance(plan_response, FixedStepPlannerResponse):
            self.agent_state = self.execute_with_steps(self.agent_state, plan_response.resource_type, plan_response.steps)
        elif isinstance(plan_response, DynamicStepPlannerResponse):
            self.agent_state = self.execute_with_steps(self.agent_state, plan_response.resource_type, plan_response.steps)

        print(f"\n --- token usage statistics: "
              f"input_statistics={self.agent_state["input_token_statistics"]}, "
              f"output_statistics={self.agent_state["output_token_statistics"]}, "
              f"total_statistics={self.agent_state["total_token_statistics"]}",
        )
        print("\n--- ✅ task complete ---")

    def execute_with_graph(self, agent_state: CodeAgentState, resource_type:str, graph: StateGraph[StateT, ContextT, InputT, OutputT]) -> CodeAgentState:
        self.agent_state["resource_type"] = resource_type
        return self.executor.execute(agent_state, graph)

    def execute_with_steps(self, agent_state: CodeAgentState, resource_type:str, steps: List[Generate]) -> CodeAgentState:
        self.agent_state["resource_type"] = resource_type
        return self.executor.execute(agent_state, steps)

