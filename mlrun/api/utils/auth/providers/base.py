import abc
import typing

import mlrun.api.schemas


class Provider(abc.ABC):
    @abc.abstractmethod
    def query_permissions(
        self,
        resource: str,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        pass

    @abc.abstractmethod
    def filter_by_permissions(
        self,
        resources: typing.List,
        opa_resource_extractor: typing.Callable,
        action: mlrun.api.schemas.AuthorizationAction,
        auth_info: mlrun.api.schemas.AuthInfo,
    ) -> typing.List:
        pass

    @abc.abstractmethod
    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: mlrun.api.schemas.AuthInfo
    ):
        pass
