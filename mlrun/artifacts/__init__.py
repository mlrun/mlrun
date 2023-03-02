# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

# Don't remove this, used by sphinx documentation
__all__ = [
    "get_model",
    "update_model",
    "Artifact",
    "ArtifactManager",
    "ArtifactProducer",
    "DatasetArtifact",
    "TableArtifact",
    "ModelArtifact",
    "PlotArtifact",
    "ChartArtifact",
    "BokehArtifact",
    "PlotlyArtifact",
    "dict_to_artifact",
    "get_artifact_cls_by_kind",
    "get_artifact_meta",
    "update_dataset_meta",
]

from .base import Artifact, get_artifact_meta
from .dataset import DatasetArtifact, TableArtifact, update_dataset_meta
from .manager import ArtifactManager, ArtifactProducer, dict_to_artifact
from .model import ModelArtifact, get_model, update_model
from .plots import BokehArtifact, ChartArtifact, PlotArtifact, PlotlyArtifact


def get_artifact_cls_by_kind(kind: str):
    artifacts_map_by_kind = {
        "artifact": Artifact,
        "dataset": DatasetArtifact,
        "table": TableArtifact,
        "model": ModelArtifact,
        "plot": PlotArtifact,
        "chart": ChartArtifact,
        "bokeh": BokehArtifact,
        "plotly": PlotlyArtifact,
    }
    return artifacts_map_by_kind[kind]
