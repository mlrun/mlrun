import http

import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.auth.verifier


def test_verify_authorization(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    authorization_verification_input = mlrun.api.schemas.AuthorizationVerificationInput(
        resource="/some-resource", action=mlrun.api.schemas.AuthorizationAction.create
    )

    def _mock_successful_query_permissions(resource, action, *args):
        assert authorization_verification_input.resource == resource
        assert authorization_verification_input.action == action

    mlrun.api.utils.auth.verifier.AuthVerifier().query_permissions = (
        _mock_successful_query_permissions
    )
    response = client.post(
        "authorization/verifications", json=authorization_verification_input.dict()
    )
    assert response.status_code == http.HTTPStatus.OK.value
