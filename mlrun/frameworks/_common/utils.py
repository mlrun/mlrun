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
MLRunInterfaceableType = TypeVar(
    "MLRunInterfaceableType"
)  # A generic object type for what can be wrapped with a framework MLRun interface (examples: xgb, xgb.XGBModel).

# Common types:
PathType = Union[str, Path]  # For receiving a path from 'pathlib' or 'os.path'.
TrackableType = Union[str, bool, float, int]  # All trackable values types for a logger.
ExtraDataType = Union[
    str, bytes, Artifact, DataItem
]  # Types available in the extra data dictionary of an artifact.
