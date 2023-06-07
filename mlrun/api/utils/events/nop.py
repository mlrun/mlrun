import typing


class NopClient:
    def emit(self, event):
        return

    def generate_project_auth_secret_created_event(
        self, username: str, secret_name: str
    ):
        return

    def generate_project_auth_secret_updated_event(
        self, username: str, secret_name: str
    ):
        return

    def generate_project_secret_created_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ):
        return

    def generate_project_secret_updated_event(
        self, project: str, secret_name: str, secret_keys: typing.List[str]
    ):
        return

    def generate_project_secret_deleted_event(self, project: str, secret_name: str):
        return
