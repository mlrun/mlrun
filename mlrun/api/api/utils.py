import collections
import re
import traceback
import typing
from hashlib import sha1
from http import HTTPStatus
from os import environ
from pathlib import Path

import kubernetes.client
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.errors
import mlrun.runtimes.pod
import mlrun.utils.helpers
from mlrun.api import schemas
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.schemas import SecretProviderName
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.logs_dir import get_logs_dir
from mlrun.api.utils.singletons.scheduler import get_scheduler
from mlrun.config import config
from mlrun.db.sqldb import SQLDB as SQLRunDB
from mlrun.k8s_utils import get_k8s_helper
from mlrun.run import import_function, new_function
from mlrun.runtimes.utils import enrich_function_from_dict
from mlrun.utils import get_in, logger, parse_versioned_object_uri


def log_and_raise(status=HTTPStatus.BAD_REQUEST.value, **kw):
    logger.error(str(kw))
    # TODO: 0.6.6 is the last version expecting the error details to be under reason, when it's no longer a relevant
    #  version can be changed to details=kw
    raise HTTPException(status_code=status, detail={"reason": kw})


def log_path(project, uid) -> Path:
    return project_logs_path(project) / uid


def project_logs_path(project) -> Path:
    return get_logs_dir() / project


def get_obj_path(schema, path, user=""):
    if path.startswith("/User/"):
        user = user or environ.get("V3IO_USERNAME", "admin")
        path = "v3io:///users/" + user + path[5:]
        schema = schema or "v3io"
    elif path.startswith("/v3io"):
        path = "v3io://" + path[len("/v3io") :]
        schema = schema or "v3io"
    elif config.httpdb.data_volume and path.startswith(config.httpdb.data_volume):
        data_volume_prefix = config.httpdb.data_volume
        if data_volume_prefix.endswith("/"):
            data_volume_prefix = data_volume_prefix[:-1]
        if config.httpdb.real_path:
            path_from_volume = path[len(data_volume_prefix) :]
            if path_from_volume.startswith("/"):
                path_from_volume = path_from_volume[1:]
            path = str(Path(config.httpdb.real_path) / Path(path_from_volume))
    if schema:
        schema_prefix = schema + "://"
        if not path.startswith(schema_prefix):
            path = f"{schema_prefix}{path}"
    if not path.startswith("v3io://") and (
        not config.httpdb.real_path
        or (config.httpdb.real_path and not path.startswith(config.httpdb.real_path))
    ):
        raise mlrun.errors.MLRunAccessDeniedError("Unauthorized path")
    return path


def get_secrets(auth_info: mlrun.api.schemas.AuthInfo):
    return {
        "V3IO_ACCESS_KEY": auth_info.data_session,
    }


def get_run_db_instance(
    db_session: Session,
):
    db = get_db()
    if isinstance(db, SQLDB):
        run_db = SQLRunDB(db.dsn, db_session)
    else:
        run_db = db.db
    run_db.connect()
    return run_db


def parse_submit_run_body(data):
    task = data.get("task")
    function_dict = data.get("function")
    function_url = data.get("functionUrl")
    if not function_url and task:
        function_url = get_in(task, "spec.function")
    if not (function_dict or function_url) or not task:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="bad JSON, need to include function/url and task objects",
        )
    return function_dict, function_url, task


def _generate_function_and_task_from_submit_run_body(
    db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data
):
    function_dict, function_url, task = parse_submit_run_body(data)
    # TODO: block exec for function["kind"] in ["", "local]  (must be a
    # remote/container runtime)

    if function_dict and not function_url:
        function = new_function(runtime=function_dict)
    else:
        if "://" in function_url:
            function = import_function(
                url=function_url, project=task.get("metadata", {}).get("project")
            )
        else:
            project, name, tag, hash_key = parse_versioned_object_uri(function_url)
            function_record = get_db().get_function(
                db_session, name, project, tag, hash_key
            )
            if not function_record:
                log_and_raise(
                    HTTPStatus.NOT_FOUND.value,
                    reason=f"runtime error: function {function_url} not found",
                )
            function = new_function(runtime=function_record)

        if function_dict:
            # The purpose of the function dict is to enable the user to override configurations of the existing function
            # without modifying it - to do that we're creating a function object from the request function dict and
            # assign values from it to the main function object
            function = enrich_function_from_dict(function, function_dict)

    # if auth given in request ensure the function pod will have these auth env vars set, otherwise the job won't
    # be able to communicate with the api
    ensure_function_has_auth_set(function, auth_info)

    # if this was triggered by the UI, we will need to attempt auto-mount based on auto-mount config and params passed
    # in the auth_info. If this was triggered by the SDK, then auto-mount was already attempted and will be skipped.
    try_perform_auto_mount(function, auth_info)

    # Validate function's service-account, based on allowed SAs for the project, if existing in a project-secret.
    process_function_service_account(function)

    mask_sensitive_data(function, auth_info)
    return function, task


async def submit_run(db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data):
    _, _, _, response = await run_in_threadpool(
        _submit_run, db_session, auth_info, data
    )
    return response


def mask_sensitive_data(function, auth_info: mlrun.api.schemas.AuthInfo):
    if not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind):
        _mask_v3io_access_key_env_var(function, auth_info)
        _mask_v3io_volume_credentials(function)


def _mask_v3io_volume_credentials(function: mlrun.runtimes.pod.KubeResource):
    """
    Go over all of the flex volumes with v3io/fuse driver of the function and try mask their access key to a secret
    """
    get_item_attribute = mlrun.runtimes.utils.get_item_name
    v3io_volume_indices = []
    # to prevent the code from having to deal both with the scenario of the volume as V1Volume object and both as
    # (sanitized) dict (it's also snake case vs camel case), transforming all to dicts
    new_volumes = []
    k8s_api_client = kubernetes.client.ApiClient()
    for volume in function.spec.volumes:
        if isinstance(volume, dict):
            if "flexVolume" in volume:
                # mlrun.platforms.iguazio.v3io_to_vol generates a dict with a class in the flexVolume field
                if not isinstance(volume["flexVolume"], dict):
                    # sanity
                    if isinstance(
                        volume["flexVolume"], kubernetes.client.V1FlexVolumeSource
                    ):
                        volume[
                            "flexVolume"
                        ] = k8s_api_client.sanitize_for_serialization(
                            volume["flexVolume"]
                        )
                    else:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"Unexpected flex volume type: {type(volume['flexVolume'])}"
                        )
            new_volumes.append(volume)
        elif isinstance(volume, kubernetes.client.V1Volume):
            new_volumes.append(k8s_api_client.sanitize_for_serialization(volume))
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unexpected volume type: {type(volume)}"
            )
    function.spec.volumes = new_volumes

    for index, volume in enumerate(function.spec.volumes):
        if volume.get("flexVolume", {}).get("driver") == "v3io/fuse":
            v3io_volume_indices.append(index)
    if v3io_volume_indices:
        volume_name_to_volume_mounts = collections.defaultdict(list)
        for volume_mount in function.spec.volume_mounts:
            # sanity
            if not get_item_attribute(volume_mount, "name"):
                logger.warning(
                    "Found volume mount without name, skipping it for volume masking username resolution",
                    volume_mount=volume_mount,
                )
                continue
            volume_name_to_volume_mounts[
                get_item_attribute(volume_mount, "name")
            ].append(volume_mount)
        for index in v3io_volume_indices:
            volume = function.spec.volumes[index]
            flex_volume = volume["flexVolume"]
            # if it's already referencing a secret, nothing to do
            if flex_volume.get("secretRef"):
                continue
            access_key = flex_volume.get("options", {}).get("accessKey")
            # sanity
            if not access_key:
                logger.warning(
                    "Found v3io fuse volume without access key, skipping masking",
                    volume=volume,
                )
                continue
            if not volume.get("name"):
                logger.warning(
                    "Found volume without name, skipping masking", volume=volume
                )
                continue
            username = _resolve_v3io_fuse_volume_access_key_matching_username(
                function, volume, volume["name"], volume_name_to_volume_mounts
            )
            if not username:
                continue
            secret_name = mlrun.api.crud.Secrets().store_auth_secret(
                mlrun.api.schemas.AuthSecretData(
                    provider=mlrun.api.schemas.SecretProviderName.kubernetes,
                    username=username,
                    access_key=access_key,
                )
            )

            del flex_volume["options"]["accessKey"]
            flex_volume["secretRef"] = {"name": secret_name}


def _resolve_v3io_fuse_volume_access_key_matching_username(
    function: mlrun.runtimes.pod.KubeResource,
    volume: dict,
    volume_name: str,
    volume_name_to_volume_mounts: dict,
) -> typing.Optional[str]:
    """
    Usually v3io fuse mount is set using mlrun.mount_v3io, which by default add a volume mount to /users/<username>, try
    to resolve the username from there
    If it's not found (user may set custom volume mounts), try to look for V3IO_USERNAME env var
    If it's not found, skip masking for this volume
    :return: the resolved username (string), none if not found
    """

    get_item_attribute = mlrun.runtimes.utils.get_item_name
    username = None
    found_more_than_one_username = False
    for volume_mount in volume_name_to_volume_mounts[volume_name]:
        # volume_mount may be an V1VolumeMount instance (object access, snake case) or sanitized dict (dict
        # access, camel case)
        sub_path = get_item_attribute(volume_mount, "subPath") or get_item_attribute(
            volume_mount, "sub_path"
        )
        if sub_path and sub_path.startswith("users/"):
            username_from_sub_path = sub_path.replace("users/", "")
            if username_from_sub_path:
                if username is not None and username != username_from_sub_path:
                    found_more_than_one_username = True
                    break
                username = username_from_sub_path
    if found_more_than_one_username:
        logger.warning(
            "Found more than one user for volume, skipping masking",
            volume=volume,
            volume_mounts=volume_name_to_volume_mounts[volume_name],
        )
        return None
    if not username:
        v3io_username = function.get_env("V3IO_USERNAME")
        if not v3io_username or not isinstance(v3io_username, str):
            logger.warning(
                "Could not resolve username from volume mount or env vars, skipping masking",
                volume=volume,
                volume_mounts=volume_name_to_volume_mounts[volume_name],
                env=function.spec.env,
            )
            return None
        username = v3io_username
    return username


def _mask_v3io_access_key_env_var(
    function: mlrun.runtimes.pod.KubeResource, auth_info: mlrun.api.schemas.AuthInfo
):
    v3io_access_key = function.get_env("V3IO_ACCESS_KEY")
    # if it's already a V1EnvVarSource or dict instance, it's already been masked
    if (
        v3io_access_key
        and not isinstance(v3io_access_key, kubernetes.client.V1EnvVarSource)
        and not isinstance(v3io_access_key, dict)
    ):
        username = None
        v3io_username = function.get_env("V3IO_USERNAME")
        if v3io_username and isinstance(v3io_username, str):
            username = v3io_username
        if not username:
            if mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required():
                # auth_info should always has username, sanity
                if not auth_info.username:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "Username is missing from auth info"
                    )
                username = auth_info.username
            else:
                logger.warning(
                    "Could not find matching username for v3io access key in env or session, skipping masking",
                )
                return
        secret_name = mlrun.api.crud.Secrets().store_auth_secret(
            mlrun.api.schemas.AuthSecretData(
                provider=mlrun.api.schemas.SecretProviderName.kubernetes,
                username=username,
                access_key=v3io_access_key,
            )
        )
        access_key_secret_key = mlrun.api.schemas.AuthSecretData.get_field_secret_key(
            "access_key"
        )
        function.set_env_from_secret(
            "V3IO_ACCESS_KEY", secret_name, access_key_secret_key
        )


def ensure_function_has_auth_set(function, auth_info: mlrun.api.schemas.AuthInfo):
    if (
        not mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind)
        and mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required()
    ):
        function: mlrun.runtimes.pod.KubeResource
        if (
            function.metadata.credentials.access_key
            == mlrun.model.Credentials.generate_access_key
        ):
            if not auth_info.access_key:
                auth_info.access_key = mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key(
                    auth_info.session
                )
            function.metadata.credentials.access_key = auth_info.access_key
        if not function.metadata.credentials.access_key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Function access key must be set (function.metadata.credentials.access_key)"
            )
        if not function.metadata.credentials.access_key.startswith(
            mlrun.model.Credentials.secret_reference_prefix
        ):
            if not auth_info.username:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Username is missing from auth info"
                )
            secret_name = mlrun.api.crud.Secrets().store_auth_secret(
                mlrun.api.schemas.AuthSecretData(
                    provider=mlrun.api.schemas.SecretProviderName.kubernetes,
                    username=auth_info.username,
                    access_key=function.metadata.credentials.access_key,
                )
            )
            function.metadata.credentials.access_key = (
                f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
            )
        else:
            secret_name = function.metadata.credentials.access_key.lstrip(
                mlrun.model.Credentials.secret_reference_prefix
            )

        access_key_secret_key = mlrun.api.schemas.AuthSecretData.get_field_secret_key(
            "access_key"
        )
        auth_env_vars = {
            mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session: (
                secret_name,
                access_key_secret_key,
            )
        }
        for env_key, (secret_name, secret_key) in auth_env_vars.items():
            function.set_env_from_secret(env_key, secret_name, secret_key)


def try_perform_auto_mount(function, auth_info: mlrun.api.schemas.AuthInfo):
    if (
        mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind)
        or function.spec.disable_auto_mount
    ):
        return
    # Retrieve v3io auth params from the caller auth info
    override_params = {}
    if auth_info.data_session or auth_info.access_key:
        override_params["access_key"] = auth_info.data_session or auth_info.access_key
    if auth_info.username:
        override_params["user"] = auth_info.username

    function.try_auto_mount_based_on_config(override_params)


def process_function_service_account(function):
    # If we're not running inside k8s, skip this check as it's not relevant.
    if not get_k8s_helper(silent=True).is_running_inside_kubernetes_cluster():
        return

    allowed_service_accounts = mlrun.api.crud.secrets.Secrets().get_project_secret(
        function.metadata.project,
        SecretProviderName.kubernetes,
        mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
            mlrun.api.crud.secrets.SecretsClientType.service_accounts, "allowed"
        ),
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )
    if allowed_service_accounts:
        allowed_service_accounts = [
            service_account.strip()
            for service_account in allowed_service_accounts.split(",")
        ]

    default_service_account = mlrun.api.crud.secrets.Secrets().get_project_secret(
        function.metadata.project,
        SecretProviderName.kubernetes,
        mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
            mlrun.api.crud.secrets.SecretsClientType.service_accounts, "default"
        ),
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )

    # Sanity check on project configuration
    if (
        default_service_account
        and allowed_service_accounts
        and default_service_account not in allowed_service_accounts
    ):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Default service account {default_service_account} is not in list of allowed "
            + f"service accounts {allowed_service_accounts}"
        )

    function.validate_and_enrich_service_account(
        allowed_service_accounts, default_service_account
    )


def _submit_run(
    db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data
) -> typing.Tuple[str, str, str, typing.Dict]:
    """
    :return: Tuple with:
        1. str of the project of the run
        2. str of the kind of the function of the run
        3. str of the uid of the run that started execution (None when it was scheduled)
        4. dict of the response info
    """
    run_uid = None
    project = None
    try:
        fn, task = _generate_function_and_task_from_submit_run_body(
            db_session, auth_info, data
        )
        if (
            mlrun.runtimes.RuntimeKinds.is_local_runtime(fn.kind)
            and not mlrun.mlconf.httpdb.jobs.allow_local_run
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Local runtimes can not be run through API (not locally)"
            )
        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db, True)
        logger.info("Submitting run", function=fn.to_dict(), task=task)
        # fn.spec.rundb = "http://mlrun-api:8080"
        schedule = data.get("schedule")
        if schedule:
            cron_trigger = schedule
            if isinstance(cron_trigger, dict):
                cron_trigger = schemas.ScheduleCronTrigger(**cron_trigger)
            schedule_labels = task["metadata"].get("labels")
            get_scheduler().create_schedule(
                db_session,
                auth_info,
                task["metadata"]["project"],
                task["metadata"]["name"],
                schemas.ScheduleKinds.job,
                data,
                cron_trigger,
                schedule_labels,
            )
            project = task["metadata"]["project"]

            response = {
                "schedule": schedule,
                "project": task["metadata"]["project"],
                "name": task["metadata"]["name"],
            }
        else:
            run = fn.run(task, watch=False)
            run_uid = run.metadata.uid
            project = run.metadata.project
            if run:
                response = run.to_dict()

    except HTTPException:
        logger.error(traceback.format_exc())
        raise
    except mlrun.errors.MLRunHTTPStatusError:
        raise
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")

    logger.info("Run submission succeeded", response=response)
    return project, fn.kind, run_uid, {"data": response}


# uid is hexdigest of sha1 value, which is double the digest size due to hex encoding
hash_len = sha1().digest_size * 2
uid_regex = re.compile(f"^[0-9a-f]{{{hash_len}}}$", re.IGNORECASE)


def parse_reference(reference: str):
    tag = None
    uid = None
    regex_match = uid_regex.match(reference)
    if not regex_match:
        tag = reference
    else:
        uid = regex_match.string
    return tag, uid
