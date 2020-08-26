# Copyright 2018 Iguazio
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

import tarfile
from base64 import b64decode, b64encode
from os import environ, path, remove
from tempfile import mktemp
from urllib.parse import urlparse

from .datastore import store_manager
from .k8s_utils import BasePod, get_k8s_helper
from .utils import logger, normalize_name, tag_image
from .config import config


def make_dockerfile(
    base_image,
    commands=None,
    src_dir=None,
    requirements=None,
    workdir="/mlrun",
    extra="",
):
    dock = "FROM {}\n".format(base_image)
    if src_dir:
        dock += "RUN mkdir -p {}\n".format(workdir)
        dock += "WORKDIR {}\n".format(workdir)
        dock += "ADD {} {}\n".format(src_dir, workdir)
        dock += "ENV PYTHONPATH {}\n".format(workdir)
    if requirements:
        dock += "RUN pip install -r {}\n".format(requirements)
    if commands:
        dock += "".join(["RUN {}\n".format(b) for b in commands])
    if extra:
        dock += extra
    print(dock)
    return dock


def make_kaniko_pod(
    context,
    dest,
    dockerfile=None,
    dockertext=None,
    inline_code=None,
    inline_path=None,
    requirements=None,
    secret_name=None,
    name="",
    verbose=False,
):

    if not dockertext and not dockerfile:
        raise ValueError("docker file or text must be specified")

    if dockertext:
        dockerfile = "/empty/Dockerfile"

    args = ["--dockerfile", dockerfile, "--context", context, "--destination", dest]
    if not secret_name:
        args.append("--insecure")
    if verbose:
        args += ["--verbosity", "debug"]

    kpod = BasePod(
        name or "mlrun-build",
        "gcr.io/kaniko-project/executor:" + config.kaniko_version,
        args=args,
        kind="build",
    )

    if secret_name:
        items = [{"key": ".dockerconfigjson", "path": ".docker/config.json"}]
        kpod.mount_secret(secret_name, "/root/", items=items)

    if dockertext or inline_code or requirements:
        kpod.mount_empty()
        commands = []
        env = {}
        if dockertext:
            commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")
            env["DOCKERFILE"] = b64encode(dockertext.encode("utf-8")).decode("utf-8")
        if inline_code:
            name = inline_path or "main.py"
            commands.append("echo ${CODE} | base64 -d > /empty/" + name)
            env["CODE"] = b64encode(inline_code.encode("utf-8")).decode("utf-8")
        if requirements:
            commands.append(
                "echo ${REQUIREMENTS} | base64 -d > /empty/requirements.txt"
            )
            env["REQUIREMENTS"] = b64encode(
                "\n".join(requirements).encode("utf-8")
            ).decode("utf-8")

        kpod.set_init_container(
            "alpine", args=["sh", "-c", "; ".join(commands)], env=env
        )

    return kpod


def upload_tarball(source_dir, target, secrets=None):
    tmpfile = mktemp(".tar.gz")
    with tarfile.open(tmpfile, "w:gz") as tar:
        tar.add(source_dir, arcname="")

    stores = store_manager.set(secrets)
    datastore, subpath = stores.get_or_create_store(target)
    datastore.upload(subpath, tmpfile)
    remove(tmpfile)


def build_image(
    dest,
    commands=None,
    source="",
    mounter="v3io",
    base_image=None,
    requirements=None,
    inline_code=None,
    inline_path=None,
    secret_name=None,
    namespace=None,
    with_mlrun=True,
    registry=None,
    interactive=True,
    name="",
    extra=None,
    verbose=False,
):

    if registry:
        dest = "{}/{}".format(registry, dest)
    elif dest.startswith("."):
        dest = dest[1:]
        if "DOCKER_REGISTRY_PORT" in environ:
            registry = urlparse(environ.get("DOCKER_REGISTRY_PORT")).netloc
        else:
            registry = environ.get("DEFAULT_DOCKER_REGISTRY")
            secret_name = secret_name or environ.get("DEFAULT_DOCKER_SECRET")
        if not registry:
            raise ValueError(
                "local docker registry is not defined, set "
                "DEFAULT_DOCKER_REGISTRY/SECRET env vars"
            )
        dest = "{}/{}".format(registry, dest)

    if isinstance(requirements, list):
        requirements_list = requirements
        requirements_path = "requirements.txt"
        if source:
            raise ValueError("requirements list only works with inline code")
    else:
        requirements_list = None
        requirements_path = requirements

    base_image = base_image or config.default_image
    if with_mlrun:
        commands = commands or []
        commands.append("pip install {}".format(config.package_path))

    if not inline_code and not source and not commands:
        logger.info("skipping build, nothing to add")
        return "skipped"

    context = "/context"
    to_mount = False
    src_dir = "."
    v3io = (
        source.startswith("v3io://") or source.startswith("v3ios://")
        if source
        else None
    )

    if inline_code:
        context = "/empty"
    elif source and "://" in source and not v3io:
        context = source
    elif source:
        if v3io:
            source = urlparse(source).path
        to_mount = True
        if source.endswith(".tar.gz"):
            source, src_dir = path.split(source)
    else:
        src_dir = None

    dock = make_dockerfile(
        base_image,
        commands,
        src_dir=src_dir,
        requirements=requirements_path,
        extra=extra,
    )

    kpod = make_kaniko_pod(
        context,
        dest,
        dockertext=dock,
        inline_code=inline_code,
        inline_path=inline_path,
        requirements=requirements_list,
        secret_name=secret_name,
        name=name,
        verbose=verbose,
    )

    if to_mount:
        # todo: support different mounters
        kpod.mount_v3io(remote=source, mount_path="/context")

    k8s = get_k8s_helper()
    kpod.namespace = k8s.resolve_namespace(namespace)

    if interactive:
        return k8s.run_job(kpod)
    else:
        pod, ns = k8s.create_pod(kpod)
        logger.info(
            'started build, to watch build logs use "mlrun watch {} {}"'.format(pod, ns)
        )
        return "build:{}".format(pod)


def build_runtime(runtime, with_mlrun, interactive=False):
    build = runtime.spec.build
    namespace = runtime.metadata.namespace
    inline = None  # noqa: F841
    if build.functionSourceCode:
        inline = b64decode(build.functionSourceCode).decode("utf-8")  # noqa: F841
    if not build.image:
        raise ValueError(
            "build spec must have a target image, set build.image = <target image>"
        )
    logger.info(f"building image ({build.image})")

    if runtime.kind == "dask":
        extra = 'ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]'
    else:
        extra = None

    name = normalize_name("mlrun-build-{}".format(runtime.metadata.name))
    base_image = tag_image(build.base_image or "mlrun/mlrun")
    if not build.base_image:
        with_mlrun = False

    status = build_image(
        build.image,
        base_image=base_image,
        commands=build.commands,
        namespace=namespace,
        # inline_code=inline,
        source=build.source,
        secret_name=build.secret,
        interactive=interactive,
        name=name,
        extra=extra,
        with_mlrun=with_mlrun,
    )
    runtime.status.build_pod = None
    if status == "skipped":
        runtime.spec.image = build.base_image
        runtime.status.state = "ready"
        return True

    if status.startswith("build:"):
        runtime.status.state = "build"
        runtime.status.build_pod = status[6:]
        return False

    logger.info("build completed with {}".format(status))
    if status in ["failed", "error"]:
        raise ValueError(" build {}!".format(status))

    local = "" if build.secret or build.image.startswith(".") else "."
    runtime.spec.image = local + build.image
    runtime.status.state = "ready"
    return True
