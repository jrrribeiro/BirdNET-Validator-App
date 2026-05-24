from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
    admin = "admin"
    validator = "validator"


class User(BaseModel):
    username: str = Field(min_length=1)
    role: Role
    active: bool = True


class Project(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid4()), min_length=8)
    project_slug: str = Field(min_length=3)
    name: str = Field(min_length=1)
    dataset_repo_id: str = Field(min_length=3)
    visibility: str = Field(default="collaborative", pattern="^(private|collaborative)$")
    owner_username: Optional[str] = None
    dataset_token: Optional[str] = None
    state_backend: str = "app_backend"
    state_repo_id: Optional[str] = None
    state_schema_version: int = 1
    state_status: str = "not_configured"
    active: bool = True


class Detection(BaseModel):
    detection_key: str = Field(min_length=16)
    audio_id: str = Field(min_length=1)
    scientific_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    source_metadata: dict[str, object] = Field(default_factory=dict)


class Validation(BaseModel):
    detection_key: str = Field(min_length=16)
    status: str = Field(min_length=1)
    corrected_species: Optional[str] = None
    notes: str = ""
    validator: str = Field(min_length=1)


