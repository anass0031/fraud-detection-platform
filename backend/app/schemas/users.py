from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(max_length=255, examples=["new.analyst@fraud.local"])
    full_name: str = Field(min_length=1, max_length=255)
    role: Literal["analyst", "admin"] = "analyst"
    password: str = Field(min_length=10, max_length=72, description="10 to 72 bytes")
