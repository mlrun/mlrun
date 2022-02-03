import asyncio
import collections
import unittest.mock

import fastapi.concurrency
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas


@pytest.mark.asyncio
async def test_calculate_pipelines_counters(db: sqlalchemy.orm.Session, monkeypatch):
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.namespace = "mlrun"
    response = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert response == collections.defaultdict(int)

    mlrun.mlconf.kfp_url = "https://somekfp-url.com"
    run_in_threadpool_mock = unittest.mock.MagicMock(return_value=asyncio.Future())
    monkeypatch.setattr(
        fastapi.concurrency, "run_in_threadpool", run_in_threadpool_mock
    )
    run_in_threadpool_mock.return_value.set_result(
        (1, 1, [{"project": "kfp", "status": mlrun.run.RunStatuses.running}])
    )
    expected_pipelines_counter = collections.defaultdict(int)
    expected_pipelines_counter.update({"kfp": 1})
    pipelines_counter = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert fastapi.concurrency.run_in_threadpool.call_count == 1
    assert pipelines_counter == expected_pipelines_counter
