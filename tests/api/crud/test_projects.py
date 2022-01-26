import asyncio
import collections
import unittest.mock

import fastapi.concurrency
import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas


@pytest.mark.asyncio
async def test_calculate_pipelines_counters(db: sqlalchemy.orm.Session,):
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.namespace = "mlrun"
    response = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert response == collections.defaultdict(int)

    mlrun.mlconf.kfp_url = "https://somekfp-url.com"
    fastapi.concurrency.run_in_threadpool = unittest.mock.MagicMock(
        return_value=asyncio.Future()
    )
    fastapi.concurrency.run_in_threadpool.return_value.set_result(
        (1, 1, [{"project": "kfp", "status": mlrun.run.RunStatuses.running}])
    )
    expected_pipelines_counter = collections.defaultdict(int)
    expected_pipelines_counter.update({"kfp": 1})
    pipelines_counter = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert fastapi.concurrency.run_in_threadpool.call_count == 1
    assert pipelines_counter == expected_pipelines_counter
