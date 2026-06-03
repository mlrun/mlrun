# Copyright 2026 Iguazio
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

"""``pipeline_context`` isolation across concurrent ``client.session()`` scopes"""

from __future__ import annotations

import asyncio

import pytest

import mlrun
from mlrun import Client, Credentials
from mlrun.projects.pipelines import pipeline_context


@pytest.fixture(autouse=True)
def _mock_dbpath(monkeypatch):
    monkeypatch.setattr(mlrun.mlconf, "dbpath", "https://mock-server")


async def test_pipeline_context_isolated_across_concurrent_client_sessions():
    """Two ``client.session()``s on two tasks must not share ``pipeline_context``.

    On the module-global singleton, the second writer clobbers the first.
    """
    client_a = Client(credentials=Credentials(token="a"))
    client_b = Client(credentials=Credentials(token="b"))

    a_assigned = asyncio.Event()
    b_assigned = asyncio.Event()

    async def task(client, project_name, mine, other):
        with client.session():
            pipeline_context.project = project_name
            mine.set()
            await other.wait()
            assert pipeline_context.project == project_name, (
                f"pipeline_context.project leaked across client sessions: "
                f"expected {project_name!r}, got {pipeline_context.project!r}"
            )

    await asyncio.gather(
        task(client_a, "proj-a", a_assigned, b_assigned),
        task(client_b, "proj-b", b_assigned, a_assigned),
    )


def test_pipeline_context_unchanged_outside_client_session():
    """Outside any ``client.session()``, ``pipeline_context`` is the legacy
    module-global; direct attribute writes persist across calls (BWC).
    """
    pipeline_context.clear(with_project=True)
    assert pipeline_context.project is None

    pipeline_context.project = "legacy-singleton"
    try:
        assert pipeline_context.project == "legacy-singleton"
    finally:
        pipeline_context.clear(with_project=True)
