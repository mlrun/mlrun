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

import unittest.mock
from unittest.mock import patch

import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fstore
from mlrun.datastore.targets import DFTarget


def test_columns_with_illegal_characters(rundb_mock):
    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "ask": [720.93, 51.96],
            "with space": [True, False],
        }
    )

    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )
    fset._run_db = rundb_mock

    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    result_df = fset.ingest(df, targets=[DFTarget()])
    assert list(result_df.columns) == ["bid_accepted", "ask", "with_space"]


def test_columns_with_illegal_characters_error():
    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "bid_accepted": [720.93, 51.96],
            "with space": [True, False],
        }
    )

    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fset.ingest(df)


def test_set_targets_with_string():
    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    fset.set_targets(["parquet", "nosql"], with_defaults=False)

    targets = fset.spec.targets

    assert len(targets) == 2

    parquet_target = None
    nosql_target = None
    for target in fset.spec.targets:
        if target.name == "parquet":
            parquet_target = target
        elif target.name == "nosql":
            nosql_target = target

    assert parquet_target.name == "parquet"
    assert parquet_target.kind == "parquet"
    assert parquet_target.partitioned

    assert nosql_target.name == "nosql"
    assert nosql_target.kind == "nosql"
    assert not nosql_target.partitioned


def test_get_default_targets_excludes_nosql_when_v3io_unavailable(monkeypatch):
    """ML-12070: nosql (V3IO-backed) should be excluded from defaults when V3IO is unavailable."""
    from mlrun.datastore.targets import get_default_targets

    monkeypatch.setattr(mlrun.mlconf.httpdb.authentication, "mode", "iguazio-v4")
    targets = get_default_targets()
    assert [t.name for t in targets] == ["parquet"]


def test_get_default_targets_includes_nosql_when_v3io_available():
    """ML-12070: nosql should be included in defaults when V3IO is available."""
    from mlrun.datastore.targets import get_default_targets

    targets = get_default_targets()
    assert [t.name for t in targets] == ["parquet", "nosql"]


def test_return_df(rundb_mock):
    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "ask": [720.93, 51.96],
            "with space": [True, False],
        }
    )
    fset._run_db = rundb_mock

    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    result_df = fset.ingest(df, targets=[DFTarget()], return_df=False)

    assert result_df is None

    result_df = fset.ingest(df, targets=[DFTarget()])

    assert isinstance(result_df, pd.DataFrame)


def test_init_featureset_graph_async_closes_resource_cache(rundb_mock):
    """Verify that the async (default) path of init_featureset_graph closes the ResourceCache."""
    mock_close_sync = unittest.mock.Mock()

    fset = fstore.FeatureSet("cache-close-test", entities=[fstore.Entity("ticker")])
    fset._run_db = rundb_mock
    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    df = pd.DataFrame({"ticker": ["GOOG", "MSFT"], "price": [100.0, 200.0]})

    with patch(
        "mlrun.feature_store.ingestion.ResourceCache.close_sync", mock_close_sync
    ):
        fset.ingest(df, targets=[DFTarget()])

    mock_close_sync.assert_called_once()


def test_init_featureset_graph_sync_closes_resource_cache():
    """Verify that the sync path of init_featureset_graph closes the ResourceCache."""
    from mlrun.feature_store.ingestion import init_featureset_graph

    mock_close_sync = unittest.mock.Mock()

    fset = fstore.FeatureSet("cache-close-test", entities=[fstore.Entity("ticker")])
    fset.spec.graph.engine = "sync"

    df = pd.DataFrame({"ticker": ["GOOG", "MSFT"], "price": [100.0, 200.0]})

    with patch(
        "mlrun.feature_store.ingestion.ResourceCache.close_sync", mock_close_sync
    ):
        init_featureset_graph(source=df, featureset=fset, namespace=None, targets=[])

    mock_close_sync.assert_called_once()


def test_run_spark_graph_closes_resource_cache():
    """Verify that run_spark_graph closes the ResourceCache."""
    mock_close_sync = unittest.mock.Mock()

    fset = fstore.FeatureSet("spark-close-test", entities=[fstore.Entity("ticker")])
    fset.spec.graph.engine = "sync"

    df = pd.DataFrame({"ticker": ["GOOG", "MSFT"], "price": [100.0, 200.0]})

    with (
        patch(
            "mlrun.feature_store.ingestion.ResourceCache.close_sync", mock_close_sync
        ),
        patch(
            "mlrun.feature_store.ingestion.create_graph_server"
        ) as mock_server_factory,
    ):
        mock_server = unittest.mock.MagicMock()
        mock_server.run.return_value = df
        mock_server_factory.return_value = mock_server

        from mlrun.feature_store.ingestion import run_spark_graph

        run_spark_graph(df, fset, namespace=None, spark=unittest.mock.MagicMock())

    mock_close_sync.assert_called_once()


def test_online_vector_service_close_closes_resource_cache():
    """Verify that OnlineVectorService.close() closes its resource cache."""
    from mlrun.feature_store.feature_vector_utils import OnlineVectorService

    mock_cache = unittest.mock.MagicMock()
    mock_graph = unittest.mock.MagicMock()

    service = OnlineVectorService(
        vector=unittest.mock.MagicMock(),
        graph=mock_graph,
        index_columns=["key"],
        resource_cache=mock_cache,
    )
    service.close()

    mock_cache.close_sync.assert_called_once()


def test_online_vector_service_close_without_cache():
    """Verify that OnlineVectorService.close() works when no cache is provided."""
    from mlrun.feature_store.feature_vector_utils import OnlineVectorService

    mock_graph = unittest.mock.MagicMock()
    service = OnlineVectorService(
        vector=unittest.mock.MagicMock(),
        graph=mock_graph,
        index_columns=["key"],
    )
    service.close()  # should not raise


def test_dask_feature_merger_close_local_client():
    """Verify that DaskFeatureMerger.close() closes a locally-created client."""
    from mlrun.feature_store.retrieval.dask_merger import DaskFeatureMerger

    mock_client = unittest.mock.MagicMock()
    merger = DaskFeatureMerger(unittest.mock.MagicMock())
    merger.client = mock_client
    merger._local_client = True

    merger.close()

    mock_client.close.assert_called_once()
    assert merger.client is None
    assert merger._local_client is False


def test_dask_feature_merger_close_external_client():
    """Verify that DaskFeatureMerger.close() does NOT close an externally-provided client."""
    from mlrun.feature_store.retrieval.dask_merger import DaskFeatureMerger

    mock_client = unittest.mock.MagicMock()
    merger = DaskFeatureMerger(unittest.mock.MagicMock(), dask_client=mock_client)

    merger.close()

    mock_client.close.assert_not_called()


def test_dask_feature_merger_close_is_idempotent():
    """Verify that calling close() twice doesn't raise."""
    from mlrun.feature_store.retrieval.dask_merger import DaskFeatureMerger

    mock_client = unittest.mock.MagicMock()
    merger = DaskFeatureMerger(unittest.mock.MagicMock())
    merger.client = mock_client
    merger._local_client = True

    merger.close()
    merger.close()

    mock_client.close.assert_called_once()


def test_offline_vector_response_close_propagates_to_merger():
    """Verify that OfflineVectorResponse.close() calls merger.close()."""
    from mlrun.feature_store.feature_vector_utils import OfflineVectorResponse

    mock_merger = unittest.mock.MagicMock()
    resp = OfflineVectorResponse(mock_merger)

    resp.close()

    mock_merger.close.assert_called_once()


def test_offline_vector_response_context_manager():
    """Verify that OfflineVectorResponse context manager calls close on exit."""
    from mlrun.feature_store.feature_vector_utils import OfflineVectorResponse

    mock_merger = unittest.mock.MagicMock()

    with OfflineVectorResponse(mock_merger):
        pass

    mock_merger.close.assert_called_once()
