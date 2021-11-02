from enum import Enum


class LogSources(Enum):
    AUTO = "auto"
    PERSISTENCY = "persistency"
    K8S = "k8s"
