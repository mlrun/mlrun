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

import copy
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
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


def echo_color(text: str, color: str | None = "auto", err: bool = False) -> None:
    """Echo text with optional color; auto‑chooses red/green when *color* is "auto"."""
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
    cwd: Path | None = None,
    input_data: str | None = None,
    debug: bool = False,
):
    """Thin wrapper around *subprocess.run* with optional debug printing."""
    if debug:
        echo_color(f"[DEBUG] {' '.join(cmd)}", color=typer.colors.MAGENTA)
    try:
        subprocess.run(
            cmd,
            check=raise_on_error,
            cwd=str(cwd) if cwd else None,
            input=input_data,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
    except subprocess.SubprocessError as exc:
        echo_color(f"[ERROR] Command failed: {' '.join(cmd)}", err=True)
        raise exc


@app.callback()
def main(ctx: typer.Context):
    if colorama is not None:
        colorama.init()
    ctx.ensure_object(dict)


def check_command_exists(cmd: str) -> bool:
    if shutil.which(cmd) is None:
        echo_color(f"[WARNING] '{cmd}' not on PATH", color=typer.colors.YELLOW)
        return False
    return True


def is_traefik_installed(debug: bool = False) -> bool:
    res = subprocess.run(
        ["kubectl", "get", "pods", "-A"], capture_output=True, text=True
    )
    if res.returncode != 0:
        return False
    return "traefik" in res.stdout.lower()


def clear_namespaces(namespace: str, debug: bool):
    echo_color("Clearing Kubernetes namespace ...")
    run_command(
        ["kubectl", "delete", "namespace", namespace], debug=debug, raise_on_error=False
    )


def ensure_namespace(namespace: str, debug: bool = False):
    try:
        run_command(["kubectl", "get", "namespace", namespace], debug=debug)
    except subprocess.CalledProcessError:
        echo_color(f"Namespace '{namespace}' not found – creating ...")
        run_command(["kubectl", "create", "namespace", namespace], debug=debug)


def setup_ingress(debug: bool):
    if is_traefik_installed(debug):
        typer.echo("Traefik already present – skipping ingress‑nginx install")
        return
    typer.echo("Installing ingress‑nginx controller ...")
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
    run_command(["helm", "repo", "update"], debug=debug)
    res = subprocess.run(
        ["helm", "status", "ingress-nginx", "-n", "ingress-nginx"],
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


def create_ingress(namespace: str, debug: bool, ingress_host: str):
    ensure_namespace(namespace, debug)
    typer.echo("Creating Ingresses ...")
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
        ("grafana", 80, "grafana"),
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
            ingress_to_apply["spec"]["rules"].append(rule)
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=yaml.dump(ingress_to_apply, sort_keys=False),
            text=True,
        )


def add_helm_repositories(debug: bool):
    echo_color("Adding Helm repos ...")
    for name, url in HELM_REPOS.items():
        run_command(["helm", "repo", "add", "--force-update", name, url], debug=debug)


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
        ["kubectl", "apply", "-f", "-"], input_data=ns_manifest.stdout, debug=debug
    )
    secret_name = "registry-credentials"
    res = subprocess.run(
        ["kubectl", "-n", namespace, "get", "secret", secret_name],
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


SEMVER_RC_REGEX = re.compile(r"^\d+\.\d+\.\d+(?:-rc\d+)?$")


def clean_version(version_str: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+(?:-rc\d+)?", version_str)
    return match.group(0) if match else version_str.strip()


def is_valid_version(version: str) -> bool:
    return bool(SEMVER_RC_REGEX.match(version))


def get_all_tags(repository: str):
    tags: list[dict[str, str]] = []
    page, per_page = 1, 100
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    while True:
        resp = requests.get(
            f"https://api.github.com/repos/{repository}/tags",
            params={"page": page, "per_page": per_page},
            timeout=30,
            headers=headers,
        )
        if resp.status_code != 200:
            echo_color(
                f"GitHub API error {resp.status_code}: {resp.text[:80]}", err=True
            )
            break
        page_tags = resp.json()
        if not page_tags:
            break
        tags.extend(page_tags)
        page += 1
    return tags


def get_latest_valid_version(repo_name: str) -> str:
    for tag in get_all_tags(repo_name):
        cleaned = clean_version(tag.get("name", ""))
        if is_valid_version(cleaned):
            echo_color(f"Using latest semver tag for {repo_name}: {cleaned}")
            return cleaned
    raise ValueError(f"No semver tag found for {repo_name}")


def setup_ce(
    server: str,
    ce_version: str,
    namespace: str,
    ce_dir: Path,
    branch: str,
    docker_creds_secret_name: str | None,
    debug: bool,
):
    if not ce_version:
        ce_version = get_latest_valid_version("mlrun/ce").replace("mlrun-ce-", "")
    add_helm_repositories(debug)
    if not ce_dir.is_dir():
        run_command(["git", "clone", REPO_URL, str(ce_dir)], debug=debug)
    if branch:
        run_command(["git", "checkout", branch], cwd=ce_dir, debug=debug)
        run_command(["git", "pull"], cwd=ce_dir, debug=debug)

    run_command(
        ["helm", "dependency", "build"], cwd=ce_dir / "charts" / "mlrun-ce", debug=debug
    )

    res = subprocess.run(
        ["helm", "status", "mlrun-admin", "-n", namespace],
        capture_output=True,
        text=True,
    )
    if "not found" in (res.stdout + res.stderr).lower():
        run_command(
            [
                "helm",
                "upgrade",
                "--install",
                "mlrun-admin",
                f"{ce_dir}/charts/mlrun-ce",
                "--namespace",
                namespace,
                "--create-namespace",
                "--devel",
                "--version",
                ce_version,
                "--values",
                f"{ce_dir}/charts/mlrun-ce/admin_installation_values.yaml",
                "--force",
            ],
            debug=False,
            cwd=ce_dir,
        )

    registry_url = server.rstrip("/")
    base_values = [
        "--namespace",
        namespace,
        "--set",
        f"global.registry.url={registry_url}",
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
        f"global.persistence.hostPath={Path.home() / 'mlrun-data'}",
        f"{ce_dir}/charts/mlrun-ce",
        "--devel",
        "--version",
        ce_version,
        "--values",
        f"{ce_dir}/charts/mlrun-ce/non_admin_cluster_ip_installation_values.yaml",
        "--set",
        "argoWorkflows.controller.metricsConfig.enabled=false",
    ]
    if docker_creds_secret_name:
        base_values.extend(
            [
                "--set",
                f"global.registry.secretName={docker_creds_secret_name}",
            ]
        )
    run_command(
        ["helm", "upgrade", "--install", "mlrun"] + base_values, debug=debug, cwd=ce_dir
    )


def upgrade_images(
    mlrun_ver: str,
    nuclio_ver: str,
    ce_dir: Path,
    user: str,
    docker_registry: str,
    arch: str,
    namespace: str,
    debug: bool,
):
    charts = ce_dir / "charts" / "mlrun-ce"
    if not charts.is_dir():
        echo_color(
            f"{charts} not found – skipping image upgrade", color=typer.colors.YELLOW
        )
        return
    if not mlrun_ver:
        mlrun_ver = get_latest_valid_version("mlrun/mlrun").lstrip("v")
    if not nuclio_ver:
        nuclio_ver = get_latest_valid_version("nuclio/nuclio").lstrip("v")
    registry_url = f"{docker_registry.rstrip('/')}/{user}"
    run_command(
        ["helm", "dependency", "build"] + (["--debug"] if debug else []),
        cwd=charts,
        debug=debug,
    )

    # nuclio maps x86_64 to amd64
    if arch == "x86_64":
        arch = "amd64"
    run_command(
        [
            "helm",
            "upgrade",
            "mlrun",
            ".",
            "--namespace",
            namespace,
            "--reuse-values",
            "--set",
            f"global.registry.url={registry_url}",
            "--set",
            "global.registry.secretName=registry-credentials",
            "--set",
            f"mlrun.api.image.tag={mlrun_ver}",
            "--set",
            f"mlrun.ui.image.tag={mlrun_ver}",
            "--set",
            f"mlrun.api.sidecars.logCollector.image.tag={mlrun_ver}",
            "--set",
            f"jupyterNotebook.image.tag={mlrun_ver}",
            "--set",
            f"nuclio.controller.image.tag={nuclio_ver}-{arch}",
            "--set",
            f"nuclio.dashboard.image.tag={nuclio_ver}-{arch}",
        ],
        cwd=charts,
        debug=debug,
    )


def add_dns_entries_to_hosts(namespace: str, target_ip: str):
    hostnames = [
        f"mlrun.{namespace}.svc.cluster.local",
        f"mlrun-api.{namespace}.svc.cluster.local",
        f"mlrun-api-chief.{namespace}.svc.cluster.local",
        f"nuclio.{namespace}.svc.cluster.local",
        f"nuclio-dashboard.{namespace}.svc.cluster.local",
        f"jupyter.{namespace}.svc.cluster.local",
        f"minio.{namespace}.svc.cluster.local",
        f"grafana.{namespace}.svc.cluster.local",
        f"kfp-ui.{namespace}.svc.cluster.local",
        f"metadata-envoy.{namespace}.svc.cluster.local",
        f"workflow-metrics.{namespace}.svc.cluster.local",
    ]
    try:
        with open("/etc/hosts", encoding="utf-8") as fp:
            lines = fp.readlines()
    except Exception as exc:
        echo_color(f"Error reading /etc/hosts: {exc}", err=True)
        return
    new_lines = [ln for ln in lines if not any(h in ln for h in hostnames)]
    new_entries = "".join(f"{target_ip} {h}\n" for h in hostnames)
    updated = "".join(new_lines) + new_entries
    try:
        subprocess.run(["sudo", "cp", "/etc/hosts", "/etc/hosts.bak"], check=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(updated)
            tmp = tf.name
        subprocess.run(["sudo", "cp", tmp, "/etc/hosts"], check=True)
        subprocess.run(["sudo", "rm", tmp], check=True)
        echo_color("/etc/hosts updated with MLRun hostnames")
    except subprocess.CalledProcessError as exc:
        echo_color(f"Failed to update /etc/hosts: {exc}", err=True)


def get_node_external_ip(debug: bool = False) -> str | None:
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
        echo_color(f"Error fetching node IP: {exc}", err=True)
    return None


def patch_mlrun_env(namespace: str):
    env_file = Path("mlrun-ce-docker.env")
    new_line = f"MLRUN_HTTPDB__DIRPATH={Path.home()}/mlrun/{namespace}/db"
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for idx, ln in enumerate(lines):
            if ln.startswith("MLRUN_HTTPDB__DIRPATH="):
                lines[idx] = new_line
                break
        else:
            lines.append(new_line)
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_file.write_text(new_line + "\n", encoding="utf-8")


def install_ce(
    user: str,
    passwd: str,
    docker_registry: str,
    ce_dir: Path,
    clear_ns: bool,
    ce_ver: str,
    mlrun_ver: str,
    nuclio_ver: str,
    branch: str,
    arch: str,
    namespace: str,
    debug: bool,
    update_hosts: bool,
    ingress_host: str,
):
    for cmd in REQUIRED_COMMANDS:
        check_command_exists(cmd)

    if clear_ns:
        clear_namespaces(namespace, debug)
    (Path.home() / "mlrun-data").mkdir(exist_ok=True)

    setup_ingress(debug)
    create_ingress(namespace, debug, ingress_host)
    create_docker_secret = user and passwd
    docker_creds_secret_name = None
    if create_docker_secret:
        docker_creds_secret_name = setup_registry_secret(
            user, passwd, docker_registry, namespace, debug
        )
    setup_ce(
        docker_registry,
        ce_ver,
        namespace,
        ce_dir,
        branch,
        docker_creds_secret_name,
        debug,
    )
    upgrade_images(
        mlrun_ver, nuclio_ver, ce_dir, user, docker_registry, arch, namespace, debug
    )
    patch_mlrun_env(namespace)

    if update_hosts:
        ip = get_node_external_ip(debug) or "127.0.0.1"
        add_dns_entries_to_hosts(namespace, ip)

    echo_color("MLRun CE installation complete! 🎉")


@app.command()
def install(
    docker_user: str = typer.Option("", help="Docker username"),
    docker_password: str = typer.Option("", help="Docker password / token"),
    docker_registry: str = typer.Option(
        ..., help="Docker registry (e.g. docker.io / registry.localhost)"
    ),
    ce_folder: Path = typer.Option(
        Path.home() / "mlrun-ce", "--ce-folder", help="Clone destination for mlrun/ce"
    ),
    clear_k8s_namespaces: bool = typer.Option(
        False, "--clear-namespaces", help="Delete namespace before install"
    ),
    ce_version: str = typer.Option(
        "", "--ce-version", help="Chart version (blank → latest)"
    ),
    mlrun_version: str = typer.Option("", "--mlrun-version", help="MLRun image tag"),
    nuclio_version: str = typer.Option("", "--nuclio-version", help="Nuclio image tag"),
    branch: str = typer.Option(
        "", "--branch", help="Git branch to checkout before upgrade"
    ),
    namespace: str = typer.Option("mlrun", "--namespace", help="Kubernetes namespace"),
    debug: bool = typer.Option(False, "--debug", help="Verbose output"),
    arch: str = typer.Option(platform.machine(), "--arch", help="CPU arch"),
    update_hosts: bool = typer.Option(
        False, "--update-hosts", help="Add hostnames to /etc/hosts"
    ),
    ingress_host: str = typer.Option(
        "", "--ingress-host", help="Ingress host suffix (e.g..platform.iguaz.io)"
    ),
):
    """High‑level installation command."""
    install_ce(
        docker_user,
        docker_password,
        docker_registry,
        ce_folder,
        clear_k8s_namespaces,
        ce_version,
        mlrun_version,
        nuclio_version,
        branch,
        arch,
        namespace,
        debug,
        update_hosts,
        ingress_host,
    )


if __name__ == "__main__":
    app()
