import unittest.mock

import pytest
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas


@pytest.mark.asyncio
async def test_calculate_pipelines_counters(db: sqlalchemy.orm.Session):
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.namespace = "mlrun"
    response = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert response == {}

    mlrun.mlconf.kfp_url = "https://somekfp-url.com"
    mlrun.api.crud.Projects()._list_pipelines_for_counters = unittest.mock.MagicMock(
        return_value=(1, 1, [{"project": "kfp", "status": mlrun.run.RunStatuses.running}]))
    expected_pipelines_counter = {"kfp": 1}
    pipelines_counter = await mlrun.api.crud.Projects()._calculate_pipelines_counters()
    assert mlrun.api.crud.Projects()._list_pipelines_for_counters.call_count == 1
    assert pipelines_counter == expected_pipelines_counter
