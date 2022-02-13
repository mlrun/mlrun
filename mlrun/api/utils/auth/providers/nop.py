import typing

import mlrun.api.schemas
import mlrun.api.utils.auth.providers.base
import mlrun.utils.singleton


class Provider(
    mlrun.api.utils.auth.providers.base.Provider,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def query_permissions(
        self,
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return True

    def filter_by_permissions(
        self,
        resources: typing.List,
        opa_resource_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List:
        return resources

    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: mlrun.api.schemas.AuthInfo
    ):
        pass
