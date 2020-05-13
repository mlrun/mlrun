from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel

from mlrun.api.schemas.user import User


# Shared properties
class ProjectBase(BaseModel):
    description: Optional[str] = None
    source: Optional[str] = None
    created: Optional[datetime] = None
    state: Optional[str] = None
    users: List[User] = []


# Properties to receive via API on creation
class ProjectCreate(ProjectBase):
    name: str
    owner: str


# Properties to receive via API on update
class ProjectUpdate(ProjectBase):
    name: Optional[str] = None
    owner: Optional[str] = None


class ProjectInDB(ProjectCreate):
    id: int = None

    class Config:
        orm_mode = True


# Additional properties to return via API
class Project(ProjectInDB):
    pass


class ProjectOut(BaseModel):
    project: Project
