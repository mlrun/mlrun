from typing import Optional, List

from pydantic import BaseModel


class EndpointIdentifiers(BaseModel):
    project: Optional[str]
    function: Optional[str]
    model: Optional[str]
    tag: Optional[str]


class EndpointMetaData(BaseModel):
    meta_data: Optional[dict]


class EndpointList(BaseModel):
    endpoints: List[EndpointIdentifiers]
