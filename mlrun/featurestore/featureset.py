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

from .model import FeatureSetStatus, FeatureSetSpec, FeatureSetMetadata
from .infer import infer_features_from_df, get_df_stats
from ..model import ModelObj


class FeatureSet(ModelObj):
    """Run template"""

    kind = "featureset"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, description=None, entities=None):
        self._spec: FeatureSetSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None

        self.spec = FeatureSetSpec(description=description, entities=entities)
        self.metadata = FeatureSetMetadata(name=name)
        self.status = None

    @property
    def spec(self) -> FeatureSetSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureSetSpec)

    @property
    def metadata(self) -> FeatureSetMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", FeatureSetMetadata)

    @property
    def status(self) -> FeatureSetStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureSetStatus)

    def infer_from_df(
        self,
        df,
        with_stats=False,
        entity_columns=None,
        timestamp_key=None,
        label_column=None,
        with_index=True,
        with_histogram=False,
        with_preview=False,
    ):
        """Infer features schema and stats from a local DataFrame"""
        infer_features_from_df(df, self._spec, entity_columns, with_index)
        if with_stats:
            get_df_stats(df, self._status, with_histogram, with_preview)
        if timestamp_key is None:
            self._spec.timestamp_key = timestamp_key
        if label_column:
            self._spec.label_column = label_column
        return self

    def add_entity(self, entity, name=None):
        self._spec.get_entities_map().update(entity, name)

    def add_feature(self, feature, name=None):
        self._spec.get_features_map().update(feature, name)

    def __getitem__(self, name):
        return self._spec.get_features_map()[name]

    def __setitem__(self, key, item):
        self._spec.get_features_map().update(item, key)

    def merge(self, other):
        pass
