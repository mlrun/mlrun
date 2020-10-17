from typing import List
import pandas as pd

from .featureset import FeatureSet


class LocalFeatureMerger:
    def __init__(self):
        pass

    def merge(self,
              entity_df,
              entity_timestamp_column: str,
              entity_primary_keys: list,
              featuresets: List[FeatureSet],
              featureset_dfs: List[pd.DataFrame],
              ):
        merged_df = entity_df
        for featureset, featureset_df in zip(featuresets, featureset_dfs):
            if featureset.spec.timestamp_key:
                merge_func = self._asof_join
            else:
                merge_func = self._join

            merged_df = merge_func(
                merged_df,
                entity_timestamp_column,
                entity_primary_keys,
                featureset,
                featureset_df,
            )
            entity_timestamp_column = featureset.spec.timestamp_key

        return merged_df

    def _asof_join(self,
                   entity_df,
                   entity_timestamp_column: str,
                   entity_primary_keys: list,
                   featureset: FeatureSet,
                   featureset_df: pd.DataFrame,
                   ):
        merged_df = pd.merge_asof(entity_df, featureset_df,
                                  left_on=entity_timestamp_column,
                                  right_on=featureset.spec.timestamp_key,
                                  left_by=entity_primary_keys,
                                  right_by=list(featureset.spec.get_entities_map().keys()))
        return merged_df

    def _join(self,
                   entity_df,
                   entity_timestamp_column: str,
                   entity_primary_keys: list,
                   featureset: FeatureSet,
                   featureset_df: pd.DataFrame,
                   ):
        merged_df = pd.merge(entity_df, featureset_df,
                             left_on=entity_primary_keys,
                             right_on=list(featureset.spec.get_entities_map().keys()))
        return merged_df

