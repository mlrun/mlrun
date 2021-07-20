import http
import typing

import fastapi
import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.errors
import mlrun.api.main
import mlrun.api.schemas
import mlrun.api.utils.background_tasks

test_router = fastapi.APIRouter()


@test_router.get(
    "/raise-python-exception",
)
def raise_python_exception(
):
    raise ValueError("Some error message")


@test_router.get(
    "/raise-mlrun-exception",
)
def raise_mlrun_exception(
):
    raise mlrun.errors.MLRunInvalidArgumentError("Some mlrun error message")


# must add it here since we're adding routes
@pytest.fixture()
def client() -> typing.Generator:
    mlrun.api.main.app.include_router(test_router, prefix="/test")
    with fastapi.testclient.TestClient(mlrun.api.main.app) as client:
        yield client


def test_exception_handling(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
):
    response_2 = client.get(f"/test/raise-mlrun-exception")
    body = response_2.json()
    bla = 1
    response = client.get(f"/test/raise-python-exception")
    bla = 2
