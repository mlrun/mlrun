#!/usr/bin/env python3
# Copyright 2025 Iguazio
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

from __future__ import annotations

import collections.abc
import copy
import enum
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import typing

import requests
import semver
import typer
import yaml

try:
    import colorama
except ImportError:
    colorama = None

app = typer.Typer(help="Manage MLRun CE installation.")

REPO_URL = "git@github.com:mlrun/ce.git"

HELM_REPOS = {
    "mlrun": "https://mlrun.github.io/ce",
    "nuclio": "https://nuclio.github.io/nuclio/charts",
    "v3io-stable": "https://v3io.github.io/helm-charts/stable",
    "minio": "https://charts.min.io/",
    "spark-operator": "https://kubeflow.github.io/spark-operator",
    "prometheus-community": "https://prometheus-community.github.io/helm-charts",
    "bitnami": "https://charts.bitnami.com/bitnami",
}

REQUIRED_COMMANDS = ["git", "helm", "kubectl"]


def echo_color(
    text: str,
    color: str | None = "auto",
    err: bool = False,
) -> None:
    """Print text to stdout/stderr with optional color; when color='auto' pick green for normal and red for errors."""
    if color == "auto":
        color = typer.colors.RED if err else typer.colors.GREEN
    if color is None:
        typer.echo(text, err=err)
    else:
        typer.echo(typer.style(text, fg=color), err=err)


def run_command(
    cmd: list[str],
    *,
    raise_on_error: bool = True,
    cwd: pathlib.Path | None = None,
    input_data: str | None = None,
    debug: bool = False,
):
    """Execute a shell command and return its stdout; optionally log the command and tolerate failures.

    On failure, prints the combined stdout/stderr of the command.
    """
    # Automatically add --debug to helm commands when debug is enabled
    if debug and cmd and cmd[0] == "helm" and "--debug" not in cmd:
        cmd = [*cmd, "--debug"]

    if debug:
        echo_color(
            f"[DEBUG] {' '.join(cmd)}",
            color=typer.colors.MAGENTA,
        )

    stdout = None
    try:
        stdout = subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=input_data,
            text=True,
            stderr=subprocess.STDOUT,
        )

    except subprocess.SubprocessError as exc:
        # stderr is merged into stdout via STDERR=STDOUT, so exc.output contains both
        combined_output = ""
        if isinstance(exc, subprocess.CalledProcessError):
            combined_output = exc.output or ""
        elif stdout:
            combined_output = stdout

        if combined_output:
            echo_color(
                f"[COMMAND OUTPUT]\n{combined_output}",
                err=True,
            )

        if raise_on_error:
            echo_color(
                f"[ERROR] Command failed: {' '.join(cmd)}",
                err=True,
            )
            raise exc
        else:
            echo_color(
                f"[WARNING] Command failed: {' '.join(cmd)}",
                err=True,
            )
    else:
        if stdout and debug:
            print(stdout)
    return stdout


@app.callback()
def main(
    ctx: typer.Context,
):
    if colorama is not None:
        colorama.init()
    ctx.ensure_object(dict)


def check_command_exists(
    cmd: str,
) -> bool:
    if shutil.which(cmd) is None:
        echo_color(
            f"[WARNING] '{cmd}' not on PATH",
            color=typer.colors.YELLOW,
        )
        return False
    return True


def is_traefik_installed(debug: bool = False) -> bool:
    res = subprocess.run(
        ["kubectl", "get", "pods", "-A"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return False
    return "traefik" in res.stdout.lower()


def clear_namespaces(
    namespace: str,
    debug: bool,
):
    res = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        echo_color(f"Namespace '{namespace}' not found – skipping delete")
        return
    echo_color("Clearing Kubernetes namespace ...")
    run_command(
        ["kubectl", "delete", "namespace", namespace],
        debug=debug,
        raise_on_error=False,
    )


def ensure_namespace(
    namespace: str,
    debug: bool = False,
):
    try:
        run_command(
            ["kubectl", "get", "namespace", namespace],
            debug=debug,
        )
    except subprocess.CalledProcessError:
        echo_color(f"Namespace '{namespace}' not found – creating ...")
        run_command(
            ["kubectl", "create", "namespace", namespace],
            debug=debug,
        )


def setup_ingress(
    debug: bool,
):
    if is_traefik_installed(debug):
        echo_color("Traefik already present – skipping ingress-nginx install")
        return
    echo_color("Installing ingress-nginx controller ...")
    run_command(
        [
            "helm",
            "repo",
            "add",
            "ingress-nginx",
            "https://kubernetes.github.io/ingress-nginx",
        ],
        debug=debug,
    )
    run_command(
        ["helm", "repo", "update"],
        debug=debug,
    )
    helm_status_cmd = [
        "helm",
        "status",
        "ingress-nginx",
        "-n",
        "ingress-nginx",
    ]
    if debug and "--debug" not in helm_status_cmd:
        helm_status_cmd.append("--debug")
    res = subprocess.run(
        helm_status_cmd,
        capture_output=True,
        text=True,
    )
    if "not found" in (res.stdout + res.stderr).lower():
        run_command(
            [
                "helm",
                "install",
                "ingress-nginx",
                "ingress-nginx/ingress-nginx",
                "--namespace",
                "ingress-nginx",
                "--create-namespace",
                "--set",
                "controller.ingressClassResource.default=true",
            ],
            debug=debug,
        )


def create_ingress(
    namespace: str,
    debug: bool,
    ingress_host: str,
):
    ensure_namespace(namespace, debug)
    echo_color("Creating Ingresses ...")
    ingress_class = "traefik" if is_traefik_installed(debug) else "nginx"
    ingress = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {"name": "mlrun-ce-ingress", "namespace": namespace},
        "spec": {"ingressClassName": ingress_class, "rules": []},
    }
    service_matrix = [
        ("mlrun-ui", 80, "mlrun"),
        ("mlrun-api", 8080, "mlrun-api"),
        ("mlrun-api-chief", 8080, "mlrun-api-chief"),
        ("nuclio-dashboard", 8070, "nuclio"),
        ("nuclio-dashboard", 8070, "nuclio-dashboard"),
        ("mlrun-jupyter", 8888, "jupyter"),
        ("minio-console", 9001, "minio"),
        ("ml-pipeline-ui", 80, "kfp-ui"),
        ("ml-pipeline", 8888, "kfp"),
        ("metadata-envoy-service", 9090, "metadata-envoy"),
        ("workflow-controller-metrics", 9091, "workflow-metrics"),
    ]
    host_suffixes = [("internal", "svc.cluster.local")]
    if ingress_host:
        host_suffixes.append(("external", ingress_host))
    for name, host_suffix in host_suffixes:
        ingress_to_apply = copy.deepcopy(ingress)
        ingress_to_apply["metadata"]["name"] += f"-{name}"
        rules = []
        for svc, port, host_prefix in service_matrix:
            host = f"{host_prefix}.{namespace}.{host_suffix}"
            rule = {
                "host": host,
                "http": {
                    "paths": [
                        {
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {"name": svc, "port": {"number": port}}
                            },
                        }
                    ]
                },
            }
            rules.append(rule)
        ingress_to_apply["spec"].get("rules", []).extend(rules)
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=yaml.dump(ingress_to_apply, sort_keys=False),
            text=True,
        )


def setup_registry_secret(
    docker_user: str,
    docker_pass: str,
    docker_registry: str,
    namespace: str,
    debug: bool,
) -> str:
    echo_color("Creating registry secret ...")
    ns_manifest = subprocess.run(
        ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"],
        capture_output=True,
        text=True,
    )
    run_command(
        ["kubectl", "apply", "-f", "-"],
        input_data=ns_manifest.stdout,
        debug=debug,
    )
    secret_name = "registry-credentials"
    res = subprocess.run(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "secret",
            secret_name,
        ],
        capture_output=True,
        text=True,
    )
    if "NotFound" in res.stderr:
        run_command(
            [
                "kubectl",
                "-n",
                namespace,
                "create",
                "secret",
                "docker-registry",
                secret_name,
                "--docker-username",
                docker_user,
                "--docker-password",
                docker_pass,
                "--docker-server",
                docker_registry,
                "--docker-email",
                f"{docker_user}@example.com",
            ],
            debug=debug,
        )
    return secret_name


class VersionSource(enum.StrEnum):
    AUTO = "auto"
    RELEASE = "release"
    TAG = "tag"


class GitHubApiPath(enum.StrEnum):
    RELEASES_LATEST = "releases/latest"
    RELEASES = "releases"
    TAGS = "tags"


SEMVER_STABLE_REGEX = re.compile(r"^\d+\.\d+\.\d+$")
SEMVER_WITH_RC_REGEX = re.compile(r"^\d+\.\d+\.\d+(?:-rc\d+)?$")
EXTRACT_VERSION_REGEX = re.compile(r"\d+\.\d+\.\d+(?:-rc\d+)?")


def clean_version(version_str: str) -> str:
    match = EXTRACT_VERSION_REGEX.search(version_str or "")
    return match.group(0) if match else (version_str or "").strip()


def is_valid_stable_version(version: str) -> bool:
    return bool(SEMVER_STABLE_REGEX.match(version))


def is_valid_semver_or_rc(version: str) -> bool:
    return bool(SEMVER_WITH_RC_REGEX.match(version))


def _github_auth_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"token {token}"} if token else {}


def _github_get_json(
    url: str,
    params: dict | None = None,
) -> tuple[int, object]:
    resp = requests.get(
        url,
        params=params,
        headers=_github_auth_headers(),
        timeout=30,
    )
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {}


def get_latest_tag(
    repository: str,
) -> str | None:
    code, data = _github_get_json(
        f"https://api.github.com/repos/{repository}/{GitHubApiPath.RELEASES_LATEST}"
    )
    if code != 200:
        echo_color(
            f"GitHub API error {code}: releases/latest {str(data)[:80]}",
            err=True,
        )
        return None
    return (data or {}).get("tag_name")


def get_all_tags(
    repository: str,
):
    tags = []
    page, per_page = 1, 100
    while True:
        code, data = _github_get_json(
            f"https://api.github.com/repos/{repository}/{GitHubApiPath.TAGS}",
            params={"page": page, "per_page": per_page},
        )
        if code != 200:
            echo_color(
                f"GitHub API error {code}: tags {str(data)[:80]}",
                err=True,
            )
            break
        page_tags = data or []
        if not page_tags:
            break
        tags.extend(page_tags)
        page += 1
    return tags


def get_all_releases(
    repository: str,
) -> list[dict]:
    """
    Fetch every release (newest first) including prereleases;
    each item may include tag_name, prerelease, published_at, etc.
    """
    releases = []
    page, per_page = 1, 100
    while True:
        code, data = _github_get_json(
            f"https://api.github.com/repos/{repository}/{GitHubApiPath.RELEASES}",
            params={
                "page": page,
                "per_page": per_page,
            },
        )
        if code != 200:
            echo_color(
                f"GitHub API error {code}: releases {str(data)[:80]}",
                err=True,
            )
            break
        page_releases = data or []
        if not page_releases:
            break
        releases.extend(page_releases)
        page += 1
    return releases


def _strip_v_prefix(
    raw_tag_text: str,
) -> str:
    if not raw_tag_text:
        return raw_tag_text
    if raw_tag_text[0] in ("v", "V"):
        return raw_tag_text[1:]
    return raw_tag_text


def _normalize_prerelease_for_numeric_order(
    raw_version_text: str,
) -> str:
    """
    Normalize labels like 'rc38' → 'rc.38' (and alpha/beta) so semver compares the numeric parts correctly.
    """
    if "-" not in raw_version_text:
        return raw_version_text

    main_part, prerelease_and_maybe_build = raw_version_text.split("-", 1)

    prerelease_part = prerelease_and_maybe_build
    build_part = None
    if "+" in prerelease_and_maybe_build:
        prerelease_part, build_part = prerelease_and_maybe_build.split("+", 1)

    lowered = prerelease_part.lower()
    normalized = prerelease_part

    def split_label_and_number(prefix: str) -> tuple[str, str] | None:
        if lowered.startswith(prefix):
            suffix = prerelease_part[len(prefix) :]
            if suffix.isdigit():
                return (prefix, suffix)
        return None

    for candidate in ("rc", "beta", "alpha"):
        split_result = split_label_and_number(candidate)
        if split_result is not None:
            label, number = split_result
            normalized = f"{label}.{number}"
            break

    if build_part is not None:
        return f"{main_part}-{normalized}+{build_part}"
    return f"{main_part}-{normalized}"


def _to_semver_version_or_none(
    raw_tag_text: str,
) -> semver.Version | None:
    if not raw_tag_text:
        return None
    stripped = _strip_v_prefix(raw_tag_text)
    normalized = _normalize_prerelease_for_numeric_order(stripped)
    try:
        return semver.Version.parse(normalized)
    except ValueError:
        return None


def get_latest_release_tag(
    repository: str,
    include_prereleases: bool,
) -> str | None:
    """
    Return the greatest semver-valid release tag, optionally excluding prereleases.

    Uses the /releases endpoint instead of /releases/latest because GitHub's "latest"
    only returns the most recently published stable release and can skip newer RCs
    or higher semantic versions. Fetching all releases ensures accurate semver ordering.
    """

    releases = get_all_releases(repository)
    if not releases:
        return None

    greatest_version = None
    greatest_original_tag = None

    for release in releases:
        tag = release.get("tag_name")
        if not tag:
            continue
        ver = _to_semver_version_or_none(tag)
        if ver is None:
            continue
        if not include_prereleases and ver.prerelease is not None:
            continue
        if greatest_version is None or ver > greatest_version:
            greatest_version = ver
            greatest_original_tag = tag

    return greatest_original_tag


def choose_first_matching_version(
    candidates: collections.abc.Iterable[str],
    *,
    allow_dev_versions: bool,
) -> str | None:
    """
    Scan candidates and return the first version matching the policy:
    - allow_dev_versions=True → X.Y.Z or X.Y.Z-rcN
    - allow_dev_versions=False → only X.Y.Z
    """
    for raw in candidates:
        name = raw or ""
        cleaned = clean_version(name)
        if allow_dev_versions:
            if is_valid_semver_or_rc(cleaned):
                return cleaned
        else:
            if is_valid_stable_version(cleaned):
                return cleaned
    return None


def resolve_from_releases(
    repo_name: str,
    allow_dev_versions: bool,
) -> str | None:
    tag = get_latest_release_tag(
        repo_name,
        include_prereleases=allow_dev_versions,
    )
    if tag:
        cleaned = clean_version(tag)
        if allow_dev_versions and is_valid_semver_or_rc(cleaned):
            echo_color(
                f"Using latest release (dev-allowed) for {repo_name}: {cleaned}",
            )
            return cleaned
        if not allow_dev_versions and is_valid_stable_version(cleaned):
            echo_color(f"Using latest release for {repo_name}: {cleaned}")
            return cleaned
    all_releases = get_all_releases(repo_name)
    chosen = choose_first_matching_version(
        (release.get("tag_name", "") for release in all_releases),
        allow_dev_versions=allow_dev_versions,
    )
    if chosen:
        echo_color(f"Using release list for {repo_name}: {chosen}")
    return chosen


def resolve_from_tags(
    repo_name: str,
    allow_dev_versions: bool,
) -> str | None:
    all_tags = get_all_tags(
        repository=repo_name,
    )
    chosen = choose_first_matching_version(
        (tag.get("name", "") for tag in all_tags),
        allow_dev_versions=allow_dev_versions,
    )
    if chosen:
        echo_color(f"Using tags list for {repo_name}: {chosen}")
    return chosen


def get_latest_valid_version(
    repo_name: str,
    *,
    allow_dev_versions: bool = False,
    version_source: typing.Literal["auto", "release", "tag"] = "auto",
) -> str:
    """
    Decide the newest acceptable version for *repo_name* based on policy and source preference.
    """
    if version_source == "release":
        chosen = resolve_from_releases(
            repo_name=repo_name,
            allow_dev_versions=allow_dev_versions,
        )
        if chosen:
            echo_color(f"Using release {chosen} for {repo_name}")
            return chosen
        echo_color(
            "Falling back to tags as releases produced no acceptable version",
            color=typer.colors.YELLOW,
        )

    if version_source == "tag":
        chosen = resolve_from_tags(
            repo_name=repo_name,
            allow_dev_versions=allow_dev_versions,
        )
        if chosen:
            echo_color(f"Using tag {chosen} for {repo_name}")
            return chosen
        echo_color(
            "Falling back to releases as tags produced no acceptable version",
            color=typer.colors.YELLOW,
        )

    chosen = resolve_from_releases(
        repo_name=repo_name,
        allow_dev_versions=allow_dev_versions,
    )
    if chosen:
        return chosen
    chosen = resolve_from_tags(
        repo_name=repo_name,
        allow_dev_versions=allow_dev_versions,
    )
    if chosen:
        echo_color(f"Using tag {chosen} for {repo_name}")
        return chosen

    raise ValueError(f"No acceptable tag found for {repo_name}")


def get_existing_helm_repos(
    debug: bool,
):
    """Return the current Helm repositories as a mapping of name→URL."""
    result = run_command(
        ["helm", "repo", "list", "--output", "json"],
        debug=debug,
    )
    return {repo["name"]: repo["url"] for repo in json.loads(result)}


def ensure_helm_repos(
    debug: bool,
):
    """Add any missing required Helm repositories, then update indices."""
    echo_color("Ensuring helm repositories ...")
    existing_repos = get_existing_helm_repos(
        debug=debug,
    )
    for name, url in HELM_REPOS.items():
        ensure_helm_repo(
            name=name,
            url=url,
            existing_repos=existing_repos,
            debug=debug,
        )
    update_helm_repos(debug)


def ensure_helm_repo(
    name: str,
    url: str,
    existing_repos: dict[str, str],
    debug: bool,
):
    """Register a Helm repository if it is missing or URL has changed."""
    if name in existing_repos:
        if existing_repos[name] == url:
            if debug:
                echo_color(
                    f"Repo '{name}' already exists with the same URL. Skipping...",
                )
            return
        else:
            echo_color(f"Adding helm repo '{name}'...")
            run_command(
                ["helm", "repo", "add", "--force-update", name, url],
                debug=debug,
            )


def update_helm_repos(
    debug: bool,
):
    """Refresh Helm repo indices."""
    run_command(
        ["helm", "repo", "update"],
        debug=debug,
    )


def _read_chart_image_versions(
    ce_dir: pathlib.Path,
) -> tuple[str | None, str | None]:
    """
    Read MLRun and Nuclio image tags from the mlrun-ce chart values.
    Returns (mlrun_version, nuclio_version) without architecture suffixes and without leading 'v'.
    """
    values_path = ce_dir / "charts" / "mlrun-ce" / "values.yaml"
    if not values_path.exists():
        return None, None
    try:
        with open(values_path, encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        mlrun_ver = (
            ((data.get("mlrun") or {}).get("api") or {}).get("image") or {}
        ).get("tag")
        nuclio_tag = (
            ((data.get("nuclio") or {}).get("controller") or {}).get("image") or {}
        ).get("tag")
        if isinstance(mlrun_ver, str):
            mlrun_ver = mlrun_ver.lstrip("v")
        if isinstance(nuclio_tag, str):
            base = nuclio_tag.split("-", 1)[0]
            nuclio_tag = base.lstrip("v")
        return mlrun_ver, nuclio_tag
    except Exception:
        return None, None


def setup_ce(
    docker_registry: str,
    ce_version: str,
    namespace: str,
    admin_namespace: str,
    ce_dir: pathlib.Path,
    branch: str,
    docker_creds_secret_name: str | None,
    debug: bool,
    mlrun_install_extra_values: dict[str, str] | None = None,
    dev_versions: bool = False,
):
    if not ce_version:
        ce_version = get_latest_valid_version(
            "mlrun/ce",
            allow_dev_versions=dev_versions,
        ).replace("mlrun-ce-", "")
    ensure_helm_repos(debug)
    if not ce_dir.is_dir():
        run_command(
            ["git", "clone", REPO_URL, str(ce_dir)],
            debug=debug,
        )
    if branch:
        run_command(
            ["git", "checkout", branch],
            cwd=ce_dir,
            debug=debug,
        )
        run_command(
            ["git", "pull"],
            cwd=ce_dir,
            debug=debug,
        )

    echo_color("Building helm dependencies ...")
    run_command(
        ["helm", "dependency", "build"],
        cwd=ce_dir / "charts" / "mlrun-ce",
        debug=debug,
    )

    helm_status_cmd = [
        "helm",
        "status",
        "mlrun-admin",
        "--namespace",
        admin_namespace,
    ]
    if debug and "--debug" not in helm_status_cmd:
        helm_status_cmd.append("--debug")
    res = subprocess.run(
        helm_status_cmd,
        capture_output=True,
        text=True,
    )
    if "not found" in (res.stdout + res.stderr).lower():
        echo_color(
            "No mlrun-admin installation found, installing mlrun-ce admin requirements",
        )
        run_command(
            [
                "helm",
                "upgrade",
                "--install",
                "mlrun-admin",
                f"{ce_dir}/charts/mlrun-ce",
                "--namespace",
                admin_namespace,
                "--create-namespace",
                "--devel",
                "--version",
                ce_version,
                "--values",
                f"{ce_dir}/charts/mlrun-ce/admin_installation_values.yaml",
                "--force",
            ],
            debug=debug,
            cwd=ce_dir,
        )

    base_values = [
        "--namespace",
        namespace,
        "--set",
        f"global.registry.url={docker_registry}",
        "--set",
        "global.externalHostAddress=mlrun.svc.cluster.local",
        "--set",
        "mlrun.api.securityContext.readOnlyRootFilesystem=false",
        "--set",
        "mlrun.api.chief.tolerations[0].key=node.kubernetes.io/disk-pressure",
        "--set",
        "mlrun.api.chief.tolerations[0].operator=Exists",
        "--set",
        "mlrun.api.chief.tolerations[0].effect=NoSchedule",
        "--set",
        "global.localEnvironment=true",
        "--set",
        "global.persistence.storageClass=hostpath",
        "--set",
        f"global.persistence.hostPath={pathlib.Path.home() / 'mlrun-data'}",
        f"{ce_dir}/charts/mlrun-ce",
        "--devel",
        "--version",
        ce_version,
        "--values",
        f"{ce_dir}/charts/mlrun-ce/non_admin_cluster_ip_installation_values.yaml",
        "--set",
        "argoWorkflows.controller.metricsConfig.enabled=false",
        "--set",
        "kube-prometheus-stack.enabled=false",
    ]
    if mlrun_install_extra_values:
        for key, value in mlrun_install_extra_values.items():
            base_values.extend(
                [
                    "--set",
                    f"{key}={value}",
                ]
            )
    if docker_creds_secret_name:
        base_values.extend(
            [
                "--set",
                f"global.registry.secretName={docker_creds_secret_name}",
            ]
        )
    echo_color("Installing MLRun CE ...")
    run_command(
        ["helm", "upgrade", "--install", "mlrun"] + base_values,
        debug=debug,
        cwd=ce_dir,
    )


def upgrade_images(
    mlrun_ver: str,
    nuclio_ver: str,
    ce_dir: pathlib.Path,
    user: str,
    docker_registry: str,
    namespace: str,
    docker_creds_secret_name: str | None,
    debug: bool,
    dev_versions: bool = False,
    use_chart_versions: bool = True,
):
    charts = ce_dir / "charts" / "mlrun-ce"
    if not charts.is_dir():
        echo_color(
            f"{charts} not found – skipping image upgrade",
            color=typer.colors.YELLOW,
        )
        return
    else:
        echo_color(f"Upgrading images in {charts} ...")

    if use_chart_versions and (not mlrun_ver or not nuclio_ver):
        chart_mlrun, chart_nuclio = _read_chart_image_versions(ce_dir)
        if not mlrun_ver and chart_mlrun:
            mlrun_ver = chart_mlrun
        if not nuclio_ver and chart_nuclio:
            nuclio_ver = chart_nuclio

    if not use_chart_versions:
        if not mlrun_ver:
            mlrun_ver = get_latest_valid_version(
                "mlrun/mlrun",
                allow_dev_versions=dev_versions,
            ).lstrip("v")
        if not nuclio_ver:
            nuclio_ver = get_latest_valid_version(
                "nuclio/nuclio",
                allow_dev_versions=dev_versions,
            ).lstrip("v")

    registry_url = f"{docker_registry}/{user}"
    run_command(
        ["helm", "dependency", "build"],
        cwd=charts,
        debug=debug,
    )

    mlrun_ver = (mlrun_ver or "").lstrip("v")

    cmd = [
        "helm",
        "upgrade",
        "mlrun",
        ".",
        "--namespace",
        namespace,
        "--reuse-values",
        "--set",
        f"global.registry.url={registry_url}",
    ]
    if mlrun_ver:
        cmd.extend(
            [
                "--set",
                f"mlrun.api.image.tag={mlrun_ver}",
                "--set",
                f"mlrun.ui.image.tag={mlrun_ver}",
                "--set",
                f"mlrun.api.sidecars.logCollector.image.tag={mlrun_ver}",
                "--set",
                f"jupyterNotebook.image.tag={mlrun_ver}",
            ]
        )
    if nuclio_ver:
        cmd.extend(
            [
                "--set",
                f"nuclio.controller.image.tag={nuclio_ver}",
                "--set",
                f"nuclio.dashboard.image.tag={nuclio_ver}",
            ]
        )
    if docker_creds_secret_name:
        cmd.extend(
            [
                "--set",
                f"global.registry.secretName={docker_creds_secret_name}",
            ]
        )
    run_command(
        cmd,
        cwd=charts,
        debug=debug,
    )


def add_dns_entries_to_hosts(
    namespace: str,
    target_ip: str,
):
    hostnames = [
        f"mlrun.{namespace}.svc.cluster.local",
        f"mlrun-api.{namespace}.svc.cluster.local",
        f"mlrun-api-chief.{namespace}.svc.cluster.local",
        f"nuclio.{namespace}.svc.cluster.local",
        f"nuclio-dashboard.{namespace}.svc.cluster.local",
        f"jupyter.{namespace}.svc.cluster.local",
        f"minio.{namespace}.svc.cluster.local",
        f"kfp-ui.{namespace}.svc.cluster.local",
        f"metadata-envoy.{namespace}.svc.cluster.local",
        f"workflow-metrics.{namespace}.svc.cluster.local",
    ]
    try:
        with open("/etc/hosts", encoding="utf-8") as fp:
            lines = fp.readlines()
    except Exception as exc:
        echo_color(
            f"Error reading /etc/hosts: {exc}",
            err=True,
        )
        return
    new_lines = [ln for ln in lines if not any(h in ln for h in hostnames)]
    new_entries = "".join(f"{target_ip} {h}\n" for h in hostnames)
    updated = "".join(new_lines) + new_entries
    try:
        subprocess.run(
            ["sudo", "cp", "/etc/hosts", "/etc/hosts.bak"],
            check=True,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
        ) as tf:
            tf.write(updated)
            tmp = tf.name
        subprocess.run(
            ["sudo", "cp", tmp, "/etc/hosts"],
            check=True,
        )
        subprocess.run(
            ["sudo", "rm", tmp],
            check=True,
        )
        echo_color("/etc/hosts updated with MLRun hostnames")
    except subprocess.CalledProcessError as exc:
        echo_color(
            f"Failed to update /etc/hosts: {exc}",
            err=True,
        )


def get_node_external_ip(
    debug: bool = False,
) -> str | None:
    try:
        res = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes = json.loads(res.stdout)
        for node in nodes.get("items", []):
            for addr in node.get("status", {}).get("addresses", []):
                if addr.get("type") == "ExternalIP":
                    if debug:
                        echo_color(
                            f"[DEBUG] External node IP: {addr['address']}",
                            color=typer.colors.MAGENTA,
                        )
                    return addr["address"]
    except Exception as exc:
        echo_color(
            f"Error fetching node IP: {exc}",
            err=True,
        )
    return None


def patch_mlrun_env(
    namespace: str,
):
    env_file = pathlib.Path("mlrun-ce-docker.env")
    new_line = f"MLRUN_HTTPDB__DIRPATH={pathlib.Path.home()}/mlrun/{namespace}/db"
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for idx, ln in enumerate(lines):
            if ln.startswith("MLRUN_HTTPDB__DIRPATH="):
                lines[idx] = new_line
                break
        else:
            lines.append(new_line)
        env_file.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    else:
        env_file.write_text(
            new_line + "\n",
            encoding="utf-8",
        )


def install_ce(
    user: str,
    passwd: str,
    docker_registry: str,
    ce_dir: pathlib.Path,
    clear_ns: bool,
    ce_ver: str,
    mlrun_ver: str,
    nuclio_ver: str,
    branch: str,
    namespace: str,
    admin_namespace: str,
    debug: bool,
    skip_update_hosts: bool,
    ingress_host: str,
    mlrun_install_extra_values: dict[str, str] | None = None,
    dev_versions: bool = False,
    use_chart_versions: bool = True,
):
    if not ce_ver:
        ce_ver = get_latest_valid_version(
            "mlrun/ce",
            allow_dev_versions=dev_versions,
        ).replace("mlrun-ce-", "")
    if use_chart_versions:
        chart_mlrun, chart_nuclio = _read_chart_image_versions(ce_dir)
        if not mlrun_ver and chart_mlrun:
            mlrun_ver = chart_mlrun
        if not nuclio_ver and chart_nuclio:
            nuclio_ver = chart_nuclio
    else:
        if not mlrun_ver:
            mlrun_ver = get_latest_valid_version(
                "mlrun/mlrun",
                allow_dev_versions=dev_versions,
            ).lstrip("v")
        if not nuclio_ver:
            nuclio_ver = get_latest_valid_version(
                "nuclio/nuclio",
                allow_dev_versions=dev_versions,
            ).lstrip("v")

    for cmd in REQUIRED_COMMANDS:
        check_command_exists(cmd)

    if clear_ns:
        clear_namespaces(namespace, debug)
    (pathlib.Path.home() / "mlrun-data").mkdir(
        exist_ok=True,
    )

    setup_ingress(debug)
    create_ingress(namespace, debug, ingress_host)
    create_docker_secret = user and passwd
    docker_creds_secret_name = None
    if create_docker_secret:
        docker_creds_secret_name = setup_registry_secret(
            user,
            passwd,
            docker_registry,
            namespace,
            debug,
        )
    setup_ce(
        docker_registry,
        ce_ver,
        namespace,
        admin_namespace,
        ce_dir,
        branch,
        docker_creds_secret_name,
        debug,
        mlrun_install_extra_values,
        dev_versions,
    )
    upgrade_images(
        mlrun_ver,
        nuclio_ver,
        ce_dir,
        user,
        docker_registry,
        namespace,
        docker_creds_secret_name,
        debug,
        dev_versions,
        use_chart_versions=use_chart_versions,
    )
    patch_mlrun_env(namespace)

    if not skip_update_hosts:
        ip = get_node_external_ip(debug) or "127.0.0.1"
        add_dns_entries_to_hosts(namespace, ip)

    echo_color("MLRun CE installation complete! 🎉")


@app.command()
def install(
    docker_user: str = typer.Option("", help="Docker username"),
    docker_password: str = typer.Option("", help="Docker password / token"),
    docker_registry: str = typer.Option(
        "registry.localhost",
        help="Docker registry (e.g. docker.io / registry.localhost)",
    ),
    ce_folder: pathlib.Path = typer.Option(
        pathlib.Path.home() / "mlrun-ce",
        "--ce-folder",
        help="Clone destination for mlrun/ce",
    ),
    clear_k8s_namespaces: bool = typer.Option(
        False,
        "--clear-namespaces",
        help="Delete namespace before install",
    ),
    ce_version: str = typer.Option(
        "",
        "--ce-version",
        help="Chart version (blank → latest)",
    ),
    mlrun_version: str = typer.Option(
        "",
        "--mlrun-version",
        help="MLRun image tag",
    ),
    nuclio_version: str = typer.Option(
        "",
        "--nuclio-version",
        help="Nuclio image tag",
    ),
    branch: str = typer.Option(
        "",
        "--branch",
        help="Git branch to checkout before upgrade",
    ),
    namespace: str = typer.Option(
        "mlrun",
        "--namespace",
        help="Kubernetes namespace",
    ),
    admin_namespace: str = typer.Option(
        "mlrun-admin",
        "--admin-namespace",
        help="Kubernetes admin namespace to install CRDs and cross tenant dependencies",
    ),
    debug: bool = typer.Option(False, "--debug", help="Verbose output"),
    skip_update_hosts: bool = typer.Option(
        True,
        "--skip-update-hosts",
        help="Add hostnames to /etc/hosts",
    ),
    ingress_host: str = typer.Option(
        "",
        "--ingress-host",
        help="Ingress host suffix (e.g..platform.iguaz.io)",
    ),
    mlrun_install_extra_values: str = typer.Option(
        None,
        "--mlrun-install-extra-values",
        help="Extra values for mlrun installation",
    ),
    dev_versions: bool = typer.Option(
        False,
        "--dev-versions",
        help="Allow non-stable tags (enables RC and non-semver tags)",
    ),
    use_latest_versions: bool = typer.Option(
        False,
        "--use-latest-versions",
        help="Query GitHub for the latest MLRun/Nuclio versions instead of using the mlrun-ce chart values",
    ),
):
    """Install or upgrade MLRun CE with configurable versions, repositories, and Kubernetes settings."""
    mlrun_install_extra_values = (
        json.loads(mlrun_install_extra_values) if mlrun_install_extra_values else None
    )

    docker_registry = docker_registry.rstrip("/")
    install_ce(
        user=docker_user,
        passwd=docker_password,
        docker_registry=docker_registry,
        ce_dir=ce_folder,
        clear_ns=clear_k8s_namespaces,
        ce_ver=ce_version,
        mlrun_ver=mlrun_version,
        nuclio_ver=nuclio_version,
        branch=branch,
        namespace=namespace,
        admin_namespace=admin_namespace,
        debug=debug,
        skip_update_hosts=skip_update_hosts,
        ingress_host=ingress_host,
        mlrun_install_extra_values=mlrun_install_extra_values,
        dev_versions=dev_versions,
        use_chart_versions=not use_latest_versions,
    )


if __name__ == "__main__":
    app()
