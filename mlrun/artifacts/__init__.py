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

# Don't remove this, used by sphinx documentation
__all__ = ['get_model', 'update_model']

from .manager import ArtifactManager, ArtifactProducer, dict_to_artifact
from .base import Artifact
from .plots import PlotArtifact, ChartArtifact
from .dataset import TableArtifact, DatasetArtifact
from .model import ModelArtifact, get_model, update_model

