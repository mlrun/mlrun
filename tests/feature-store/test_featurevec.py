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
#


from datetime import datetime
from unittest import mock

from mlrun.feature_store.common import RunConfig
from mlrun.feature_store.feature_vector import FeatureVector, FixedWindowType
from mlrun.model import DataTargetBase


@mock.patch("mlrun.feature_store.api._get_online_feature_service")
def test_get_online_feature_service(mock_get_online_service):
    fv = FeatureVector()

    test_run_config = RunConfig()
    test_fixed_window_type = FixedWindowType.LastClosedWindow
    test_impute_policy = {"policy": "mean"}
    test_update_stats = True
    test_entity_keys = ["key1", "key2"]

    fv.get_online_feature_service(
        run_config=test_run_config,
        fixed_window_type=test_fixed_window_type,
        impute_policy=test_impute_policy,
        update_stats=test_update_stats,
        entity_keys=test_entity_keys,
    )

    mock_get_online_service.assert_called_once_with(
        fv,
        test_run_config,
        test_fixed_window_type,
        test_impute_policy,
        test_update_stats,
        test_entity_keys,
    )


@mock.patch("mlrun.feature_store.api._get_offline_features")
def test_get_offline_features(mock_get_offline_features):
    fv = FeatureVector()

    # Define your test inputs
    test_entity_rows = None
    test_entity_timestamp_column = "timestamp"
    test_target = DataTargetBase()  # Assuming DataTargetBase is a valid class
    test_run_config = RunConfig()  # Assuming RunConfig is a valid class
    test_drop_columns = ["col1", "col2"]
    test_start_time = "2021-01-01"
    test_end_time = datetime.now()
    test_with_indexes = True
    test_update_stats = False
    test_engine = "test_engine"
    test_engine_args = {"arg1": "value1"}
    test_query = "SELECT * FROM table"
    test_order_by = "col1"
    test_spark_service = "test_spark_service"
    test_timestamp_for_filtering = {"col1": "2021-01-01"}
    additional_filters = [("x", "=", 3)]

    fv.get_offline_features(
        entity_rows=test_entity_rows,
        entity_timestamp_column=test_entity_timestamp_column,
        target=test_target,
        run_config=test_run_config,
        drop_columns=test_drop_columns,
        start_time=test_start_time,
        end_time=test_end_time,
        with_indexes=test_with_indexes,
        update_stats=test_update_stats,
        engine=test_engine,
        engine_args=test_engine_args,
        query=test_query,
        order_by=test_order_by,
        spark_service=test_spark_service,
        timestamp_for_filtering=test_timestamp_for_filtering,
        additional_filters=additional_filters,
    )
    mock_get_offline_features.assert_called_once_with(
        fv,
        test_entity_rows,
        test_entity_timestamp_column,
        test_target,
        test_run_config,
        test_drop_columns,
        test_start_time,
        test_end_time,
        test_with_indexes,
        test_update_stats,
        test_engine,
        test_engine_args,
        test_query,
        test_order_by,
        test_spark_service,
        test_timestamp_for_filtering,
        additional_filters,
    )
