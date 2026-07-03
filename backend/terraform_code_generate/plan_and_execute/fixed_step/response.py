from typing import List
from pydantic import  Field

from backend.terraform_code_generate.plan_and_execute.response import PlannerResponse

class FixedStepPlannerResponse(PlannerResponse):

    steps: List[str] = Field(description="The steps to be executed.")