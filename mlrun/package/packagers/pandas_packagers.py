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
import pandas as pd

from mlrun.artifacts import DatasetArtifact
from mlrun.datastore import DataItem

from ..utils import ArtifactType
from .default_packager import DefaultPackager


class PandasDataFramePackager(DefaultPackager):
    PACKABLE_OBJECT_TYPE = pd.DataFrame
    DEFAULT_ARTIFACT_TYPE = ArtifactType.DATASET

    @classmethod
    def pack_dataset(cls, obj: pd.DataFrame, key: str, fmt: str = "parquet"):
        return DatasetArtifact(key=key, df=obj, format=fmt), None

    @classmethod
    def unpack_dataset(cls, data_item: DataItem):
        return data_item.as_df()


class PandasSeriesPackager(PandasDataFramePackager):
    PACKABLE_OBJECT_TYPE = pd.Series
    DEFAULT_ARTIFACT_TYPE = ArtifactType.DATASET

    @classmethod
    def pack_dataset(cls, obj: pd.Series, key: str, fmt: str = "parquet"):
        pass

    @classmethod
    def unpack_dataset(cls, data_item: DataItem):
        data_frame = super().unpack_dataset(data_item=data_item)
