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

from typing import List
import pandas as pd


class LocalFeatureMerger:
    def __init__(self):
        self._result_df = None

    def merge(
        self,
        entity_df,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: List[pd.DataFrame],
    ):
        merged_df = entity_df
        if entity_df is None:
            merged_df = featureset_dfs.pop(0)
            featureset = featuresets.pop(0)
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

        for featureset, featureset_df in zip(featuresets, featureset_dfs):
            if featureset.spec.timestamp_key:
                merge_func = self._asof_join
            else:
                merge_func = self._join

            merged_df = merge_func(
                merged_df, entity_timestamp_column, featureset, featureset_df,
            )

        self._result_df = merged_df

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df: pd.DataFrame,
    ):
        indexes = list(featureset.spec.entities.keys())
        merged_df = pd.merge_asof(
            entity_df,
            featureset_df,
            left_on=entity_timestamp_column,
            right_on=featureset.spec.timestamp_key,
            by=indexes,
        )
        return merged_df

    def _join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df: pd.DataFrame,
    ):
        indexes = list(featureset.spec.entities.keys())
        merged_df = pd.merge(entity_df, featureset_df, on=indexes)
        return merged_df

    def get_status(self):
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "ready"

    def get_df(self):
        return self._result_df
