from pathlib import Path
from typing import TypeVar, Union

from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem

# Generic types:
ModelType = TypeVar(
    "ModelType"
)  # A generic model type in a handler / interface (examples: tf.keras.Model, torch.Module).
IOSampleType = TypeVar(
    "IOSampleType"
)  # A generic inout / output samples for reading the inputs / outputs properties.

# Common types:
PathType = Union[str, Path]  # For receiving a path from 'pathlib' or 'os.path'
ExtraDataType = Union[
    str, bytes, Artifact, DataItem
]  # Types available in the extra data dictionary of an artifact
