from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Extra


class ObjectMetadata(BaseModel):
    name: str
    tag: Optional[str]
    labels: Optional[dict]
    updated: Optional[datetime]
    uid: Optional[str]

    class Config:
        extra = Extra.allow
