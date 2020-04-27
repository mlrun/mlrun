from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel

from .user import User


# Shared properties
class ProjectBase(BaseModel):
    name: str
    owner: str
    description: Optional[str] = None
    source: Optional[str] = None
    created: Optional[datetime] = None
    state: Optional[str] = None
    users: List[User] = []


# Properties to receive via API on creation
class ProjectCreate(ProjectBase):
    pass


# Properties to receive via API on update
class ProjectUpdate(ProjectBase):
    pass


class ProjectInDB(ProjectBase):
    id: int = None

    class Config:
        orm_mode = True


# Additional properties to return via API
class Project(ProjectInDB):
    pass
