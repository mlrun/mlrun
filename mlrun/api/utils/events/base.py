import abc


class BaseEventClient:
    @abc.abstractmethod
    def emit(self, event):
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
