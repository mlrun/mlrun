from typing import Dict, Union
from enum import Enum


class ArtifactType(Enum):
    """
    Possible artifact types to log using the MLRun `context` decorator.
    """

    # Types:
    DATASET = "dataset"
    DIRECTORY = "directory"
    FILE = "file"
    OBJECT = "object"
    PLOT = "plot"
    RESULT = "result"

    # Constants:
    DEFAULT = RESULT


class PackageKeys:
    KEY = "key"
    ARTIFACT_TYPE = "artifact_type"


class ArtifactKeys(PackageKeys):
    DB_KEY = "db_key"


class ResultKeys(PackageKeys):
    pass


class DatasetKeys(ArtifactKeys):
    FORMAT = "format"
    EXTRA_DATA = "extra_data"


class ModelKeys(ArtifactKeys):
    EXTRA_DATA = "extra_data"
    INPUTS = "inputs"
    OUTPUTS = "outputs"


TypeHint = Union[str, type]
LogHint = Union[str, Dict[str, str]]
