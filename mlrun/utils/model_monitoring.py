import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import mlrun
from mlrun.config import config
from mlrun.platforms.iguazio import parse_v3io_path
from mlrun.utils import parse_versioned_object_uri


@dataclass
class FunctionURI:
    project: str
    function: str
    tag: Optional[str] = None
    hash_key: Optional[str] = None

    @classmethod
    def from_string(cls, function_uri):
        project, uri, tag, hash_key = parse_versioned_object_uri(function_uri)
        return cls(
            project=project, function=uri, tag=tag or None, hash_key=hash_key or None,
        )


@dataclass
class VersionedModel:
    model: str
    version: Optional[str]

    @classmethod
    def from_string(cls, model):
        try:
            model, version = model.split(":")
        except ValueError:
            model, version = model, None

        return cls(model, version)


@dataclass
class EndpointUID:
    project: str
    function: str
    function_tag: str
    function_hash_key: str
    model: str
    model_version: str
    uid: Optional[str] = None

    def __post_init__(self):
        function_ref = (
            f"{self.function}_{self.function_tag or self.function_hash_key or 'N/A'}"
        )
        versioned_model = f"{self.model}_{self.model_version or 'N/A'}"
        unique_string = f"{self.project}_{function_ref}_{versioned_model}"
        self.uid = hashlib.sha1(unique_string.encode("utf-8")).hexdigest()

    def __str__(self):
        return self.uid


def create_model_endpoint_id(function_uri: str, versioned_model: str):
    function_uri = FunctionURI.from_string(function_uri)
    versioned_model = VersionedModel.from_string(versioned_model)

    if (
        not function_uri.project
        or not function_uri.function
        or not versioned_model.model
    ):
        raise ValueError("Both function_uri and versioned_model have to be initialized")

    uid = EndpointUID(
        function_uri.project,
        function_uri.function,
        function_uri.tag,
        function_uri.hash_key,
        versioned_model.model,
        versioned_model.version,
    )

    return uid


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = parse_v3io_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def set_project_model_monitoring_credentials(
    access_key: str, project: Optional[str] = None
):
    """ Set the credentials that will be used by the project's model monitoring
    infrastructure functions.
    The supplied credentials must have data access

    :param access_key: Model Monitoring access key for managing user permissions.
    :param project: The name of the model monitoring project.
    """
    mlrun.get_run_db().create_project_secrets(
        project=project or config.default_project,
        provider=mlrun.api.schemas.SecretProviderName.kubernetes,
        secrets={"MODEL_MONITORING_ACCESS_KEY": access_key},
    )


class EndpointType(IntEnum):
    NODE_EP = 1  # end point that is not a child of a router
    ROUTER = 2  # endpoint that is router
    LEAF_EP = 3  # end point that is a child of a router
