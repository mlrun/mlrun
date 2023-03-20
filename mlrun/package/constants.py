from typing import Dict, Union


class ArtifactType:
    # TODO: Remove
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


class ArtifactTypes:
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


TypeHint = Union[str, type]
LogHint = Union[str, Dict[str, str]]
