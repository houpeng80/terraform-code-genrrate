from backend.terraform_code_generate.agents.code_agent.data_source_agent.prompt import SKILLS as DATA_SOURCE_SKILLS
from backend.terraform_code_generate.agents.code_agent.resource_agent.prompt import SKILLS as RESOURCE_SKILLS

SKILLS = {
    "resource": RESOURCE_SKILLS,
    "data_source":DATA_SOURCE_SKILLS
}