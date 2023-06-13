import abc
import typing

import mlrun.common.schemas


class BaseEventClient:
    @abc.abstractmethod
    def emit(self, event):
        pass

    def generate_project_auth_secret_event(
        self,
        username: str,
        secret_name: str,
        action: mlrun.common.schemas.AuthSecretEventActions,
    ):
        """
        Generate a project auth secret event
        :param username:  username
        :param secret_name:  secret name
        :param action: preformed action
        :return: event object to emit
        """
        pass

    @abc.abstractmethod
    def generate_project_auth_secret_created_event(
        self, username: str, secret_name: str
    ):
        pass

    @abc.abstractmethod
    def generate_project_auth_secret_updated_event(
        self, username: str, secret_name: str
    ):
        pass

    @abc.abstractmethod
    def generate_project_secret_event(
        self,
        project: str,
        secret_name: str,
        secret_keys: typing.List[str] = None,
        action: mlrun.common.schemas.SecretEventActions = mlrun.common.schemas.SecretEventActions.created,
    ):
        """
        Generate a project secret event
        :param project: project name
        :param secret_name: secret name
        :param secret_keys: secret keys, optional, only relevant for created/updated events
        :param action: preformed action
        :return: event object to emit
        """
        pass

    @abc.abstractmethod
    def generate_project_secret_created_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ):
        pass

    @abc.abstractmethod
    def generate_project_secret_updated_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ):
        pass

    @abc.abstractmethod
    def generate_project_secret_deleted_event(self, project: str, secret_name: str):
        pass
