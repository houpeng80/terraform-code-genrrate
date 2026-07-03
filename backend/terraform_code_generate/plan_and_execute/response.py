from pydantic import BaseModel, Field


class PlannerResponse(BaseModel):
    """Detail information for a resource."""
    resource_type: str = Field(description="The generated resource type， it can be resource or data_source.")