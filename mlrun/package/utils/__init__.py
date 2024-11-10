# Copyright 2023 Iguazio
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

from ._archiver import ArchiveSupportedFormat
from ._formatter import StructFileSupportedFormat
from ._pickler import Pickler
from ._supported_format import SupportedFormat
from .log_hint_utils import LogHintKey, LogHintUtils
from .type_hint_utils import TypeHintUtils

# The default pickle module to use for pickling objects:
DEFAULT_PICKLE_MODULE = "cloudpickle"
# The default archive format to use for archiving directories:
DEFAULT_ARCHIVE_FORMAT = ArchiveSupportedFormat.ZIP
# The default struct file format to use for savings python struct objects (dictionaries and lists):
DEFAULT_STRUCT_FILE_FORMAT = StructFileSupportedFormat.JSON


class ArtifactType:
    """
    Possible artifact types to pack objects as and log using a `mlrun.Packager`.
    """

    OBJECT = "object"
    PATH = "path"
    FILE = "file"
    DATASET = "dataset"
    MODEL = "model"
    PLOT = "plot"
    RESULT = "result"


class DatasetFileFormat:
    """
    All file format for logging objects as `DatasetArtifact`.
    """

    CSV = "csv"
    PARQUET = "parquet"
