from datetime import datetime
from typing import Optional, List, Union

from pydantic import BaseModel, Extra


class ProjectPatch(BaseModel):
    description: Optional[str] = None
    source: Optional[str] = None
    state: Optional[str] = None
    owner: Optional[str] = None
    created: Optional[datetime] = None

    class Config:
        extra = Extra.allow


class Project(ProjectPatch):
    name: str


class ProjectRecord(Project):
    id: int = None

    class Config:
        orm_mode = True


class ProjectsOutput(BaseModel):
    # use the full query param to control whether the full object will be returned or only the names
    projects: List[Union[Project, str]]
