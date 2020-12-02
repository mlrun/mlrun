from dataclasses import dataclass, Field
from datetime import datetime
from typing import Optional, List


@dataclass()
class EndpointKey:
    project: str
    function: str
    model: str
    tag: str
    model_class: Optional[str] = None,
    hash: Optional[str] = None

    def __post_init__(self):
        self.hash: str = f"{self.project}_{self.function}_{self.model}_{self.tag}"

    def __str__(self):
        return self.hash


#TODO Compute left over EndpointState variables
@dataclass
class EndpointState:
    endpoint_key: EndpointKey
    event_count: int = 0
    first_event: Optional[datetime] = None
    last_event: Optional[datetime] = None
    last_flushed: Optional[datetime] = None
    predictions: List[dict] = Field(default_factory=list)
    alert_count: int = 0
    # "expected_feature_values: ...
    # "actual_feature_values": ...
    # "drift_status": ...
    # "accuracy": ...
