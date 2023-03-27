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
#
class ArtifactType:
    """
    Possible artifact types to pack objects as and log using a `mlrun.Packager`.
    """

    DATASET = "dataset"
    DIRECTORY = "directory"
    FILE = "file"
    OBJECT = "object"
    PLOT = "plot"
    RESULT = "result"


class LogHintKey:
    """
    Known keys for a log hint to have.
    """

    KEY = "key"
    ARTIFACT_TYPE = "artifact_type"
    EXTRA_DATA = "extra_data"
    METRICS = "metrics"
