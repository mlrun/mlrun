from datetime import datetime
from typing import Optional, List, Union

from pydantic import BaseModel, Extra


class ProjectUpdate(BaseModel):
    description: Optional[str] = None
    source: Optional[str] = None
    state: Optional[str] = None
    owner: Optional[str] = None

    class Config:
        extra = Extra.allow


# Properties to receive via API on creation
class ProjectCreate(ProjectUpdate):
    name: str


class ProjectRecord(ProjectCreate):
    id: int = None
    created: Optional[datetime] = None

    class Config:
        orm_mode = True


# Additional properties to return via API
class Project(ProjectRecord):
    pass


class ProjectsOutput(BaseModel):
    # use the full query param to control whether the full object will be returned or only the names
    projects: List[Union[Project, str]]
