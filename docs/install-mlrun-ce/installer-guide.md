(mlrun-ce-installer-guide)=
# MLRun CE installer guide

The MLRun CE installer (`mlrun-ce-installer`) is a command-line tool that installs the MLRun CE Helm chart for you. 

In one command it creates the namespace and the registry pull secret, runs `helm upgrade --install`, 
and prints the URLs of the installed services. It can also upgrade and uninstall the release.

It installs the **published** chart from `https://mlrun.github.io/ce`, and is additive: installing by hand with `helm install`, 
as described in [Install MLRun CE on Kubernetes](./kubernetes-install.md), still works as before.

```{admonition} Note
The installer's source and its full reference live in the [mlrun/ce repository](https://github.com/mlrun/ce/tree/development/scripts). 
This page walks through the common workflows.
```

**In this section**
- [Prerequisites](#prerequisites)
- [Get the installer](#get-the-installer)
- [Install](#install)
- [Configure the installation](#configure-the-installation)
- [Uninstall](#uninstall)
- [Flag reference](#flag-reference)

## Prerequisites

| Tool | Notes |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Supplies the Python interpreter and the installer's dependencies. Nothing to `pip install`. Install with `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Helm](https://helm.sh/docs/intro/install/) | Version 4.1 or newer. The installer refuses an older Helm |
| [kubectl](https://kubernetes.io/docs/tasks/tools/) | Pointing at the target cluster |

Docker is needed only for a local cluster (Docker Desktop, kind, minikube) and for pushing your own images to a `--local-registry`. 

The installer itself uses the `docker` CLI only for a best-effort registry login check, which is skipped when Docker is absent.

See also the [Kubernetes prerequisites](./kubernetes-install.md#prerequisites). For Docker Desktop specifics (memory, port-forwarding, Apple Silicon), see [Local install on Docker Desktop](https://github.com/mlrun/ce/blob/development/scripts/README.md#local-install-on-docker-desktop) in the installer README.

## Get the installer

**Without a clone** (recommended). Pin a release tag from the [releases page](https://github.com/mlrun/ce/releases). 

Replace `<chart-version>` with the latest GA chart version (not an `-rc`), currently **{{ cechartversion }}**:

```bash
CE_VERSION=<chart-version>       # the chart version to install
CE_TAG="mlrun-ce-${CE_VERSION}"  # the matching release tag
uvx --from "git+https://github.com/mlrun/ce@${CE_TAG}#subdirectory=scripts" \
  mlrun-ce-installer install --ce-version "${CE_VERSION}"
```

```{admonition} The tag pins the installer, not the chart
`CE_TAG` selects which version of the installer code runs. It does **not** select the chart. 
Without `--ce-version` (or the `CE_VERSION` environment variable), the installer installs the newest *stable* published chart, 
which can be newer than the tag. Rerunning the same command later can therefore upgrade your release. 
To pin both, use a tag for the installer **and** pass `--ce-version` for the chart, as in the example above.
```

Tags older than the installer have no `scripts/` directory and fail. This path needs access to GitHub and PyPI, 
and resolves the installer's dependencies (typer, click, rich, pyyaml) within the ranges in `scripts/pyproject.toml`. 
A clone uses the exact versions pinned in `scripts/install.py.lock`.

**As a command**:

```bash
uv tool install "git+https://github.com/mlrun/ce@${CE_TAG}#subdirectory=scripts"
mlrun-ce-installer version   # prints the chart version this installer ships with (the tag's version), not necessarily the chart that will be installed
mlrun-ce-installer install --ce-version "${CE_VERSION}"
```

**From a clone** of [mlrun/ce](https://github.com/mlrun/ce):

```bash
./scripts/install.py
```

The examples below use `./scripts/install.py`. The `mlrun-ce-installer` command takes the same arguments.

The commands are `install` (the default), `uninstall`, `version` and `help`. 
Flags can also be passed with no command - `-h`/`--help` and `-v`/`--version` are also accepted.

## Install

### Interactive

```bash
./scripts/install.py
```

The installer asks for the registry credentials, the external host address (the address you use to reach the cluster's NodePorts), and the registry URL. 
Nothing on the cluster changes while you answer. It then shows the target and asks before its first write:

```
[INFO] About to install:
[INFO]   Cluster:   docker-desktop (https://127.0.0.1:6443)
[INFO]   Namespace: mlrun
[INFO]   Release:   mlrun-ce
[INFO]   Chart:     mlrun-ce/mlrun-ce (newest stable)
[INFO]   Registry:  index.docker.io/myuser (pull secret 'registry-credentials')
[INFO]   Address:   host.docker.internal
Proceed? [Y/n]:
```

Check the cluster line: it is your kubeconfig's current context, or the one you set with `--kube-context`. 
Answering `n`, or pressing Ctrl-C, leaves the cluster untouched.

### Choose a registry

MLRun builds an image for some of the functions you deploy and pushes it to a registry. The installer stores the registry credentials as a pull secret named `registry-credentials`.
- **Docker Hub**: your username, an [access token](https://docs.docker.com/security/for-developers/access-tokens/) as the password, and `index.docker.io/<username>` as the registry URL.
- **No registry account**: use `--local-registry` to run one in the cluster, see [Local registry](#local-registry).
- **A secret you created yourself**: pass `--skip-secret`. The installer checks that it exists and never modifies it.

### Non-interactive (CI)

`--non-interactive`, or `CI=true`, never prompts. A missing required value exits with code 1 before anything changes.

```bash
export CI=true
export REGISTRY_USERNAME=myuser
export REGISTRY_PASSWORD_FILE=/run/secrets/registry-password   # or REGISTRY_PASSWORD
export REGISTRY_SERVER=https://index.docker.io/v1/
export REGISTRY_EMAIL=me@example.com
export REGISTRY_URL=index.docker.io/myuser
export EXTERNAL_HOST_ADDRESS=localhost
./scripts/install.py
```

The password is never passed to the helm, kubectl or docker processes as a command-line argument, and it never reaches their environment. 
The installer passes it to `kubectl apply` and `docker login` on stdin.

### Dry run

```bash
./scripts/install.py --dry-run
```

Runs every check, prints the plan, and renders the chart with `helm upgrade --install --dry-run=server`. Nothing is created. 

The run ends by listing the files to review: the debug log (the rendered manifests with Secret values redacted, and every command), `mlrun-ce-values.yaml`, 
and, on an upgrade, the release's current values and the values the upgrade changes.

### Pre-install checks

These run before anything is created, dry runs included.

- **Blocking**: Helm older than 4.1, and no default StorageClass. Both are reported before the installer exits, so you can fix them in one pass.
- **Warnings**: Kubernetes older than 1.34 (or older than `MIN_K8S_VERSION`, if set); a failed registry login; no IngressClass matching `--enable-ingress`; less than 8Gi allocatable memory or ephemeral storage in total; arm64 nodes (MLRun publishes some images for amd64 only).
- **NodePorts**: if a default chart NodePort is held by another Service, the installer moves it to the next free port and warns you. The new port is used in all printed URLs. A busy NodePort that you set yourself in a `-f` file is an error, not moved.

It is skipped with `--local-registry`, `--skip-secret`, `--dry-run`, when only `-f` is used, and when the `docker` CLI is not available.

Use `--skip-validators` to skip all of them.

### After installation

The run ends with a table of the services and where to reach them:

```
                          MLRun CE - Access URLs
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ SERVICE            ┃ URL                        ┃ CREDENTIALS          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Jupyter UI         │ host.docker.internal:30040 │                      │
│ Nuclio UI          │ host.docker.internal:30050 │                      │
│ MLRun UI           │ host.docker.internal:30060 │                      │
│ MLRun API          │ host.docker.internal:30070 │                      │
│ ...                │ ...                        │                      │
└────────────────────┴────────────────────────────┴──────────────────────┘
```

Open the addresses with `http://`. Disabled components are left out. With `--enable-ingress`, each UI shows its ingress host. 
On Docker Desktop the NodePorts may not be published to your machine. If the addresses don't open, forward the ports with `kubectl port-forward`, 
as described in the [installer README](https://github.com/mlrun/ce/blob/development/scripts/README.md#local-install-on-docker-desktop).

## Configure the installation

### Precedence

Highest first: **flag > environment variable > `ce-config.yaml` > `-f` values file > chart defaults**.

Everything the installer resolves is written to `mlrun-ce-values.yaml` (`GENERATED_VALUES_FILE`) and passed to Helm as the last `--values` file, 
after your `-f` file, so it wins for the keys it sets. Ingress hosts and classes are the exception: one set in your `-f` file is kept. 
Chart values the installer does not manage (resources, replicas, and so on) come only from your `-f` file.

If `mlrun-ce-values.yaml` already exists and the run would change it, the installer lists the differences and asks you to 
**merge** (keep the keys only the old file sets, take this run's value for the rest), **override** (the default), or **cancel** (stop with nothing changed). 
The old file is kept as `mlrun-ce-values.yaml.bak`. With `--non-interactive` or `CI=true` there is no question: the installer overrides.

```{admonition} Values from a previous installation can carry over
**Merge** keeps the keys that only the old file sets, so settings from the earlier installation, including values you added by hand, stay in effect even if this run no longer needs them. 
Merged keys can be stale, so check the list before merging.
```

### Config file

A config file makes every run repeatable. Start from an example in the [mlrun/ce repository](https://github.com/mlrun/ce/tree/development/scripts), 
which lists every key with the flag or variable it stands for:

```bash
cp scripts/ce-config.local.yaml.example ce-config.yaml    # Docker Desktop, kind, minikube
cp scripts/ce-config.remote.yaml.example ce-config.yaml   # any other cluster
./scripts/install.py --config ce-config.yaml
```

### Your own Helm values

```bash
./scripts/install.py -f my-values.yaml
```

Used alone, `-f` is passed to Helm as is. No pull secret is created and no registry questions are asked, 
so the file must set `global.registry.*` and the secret it names must already exist. 
Combine it with `--config` to have the installer handle the secret and registry as usual. In that case the values the installer resolves win over the same keys in your file.

### Optional components

```bash
./scripts/install.py --disable-system-monitoring --disable-spark --disable-mpi --disable-model-monitoring
```

Each flag leaves one optional component out of the installation. Use only the flags for the components you do not need:

| Flag | What it leaves out | What you lose |
|---|---|---|
| `--disable-system-monitoring` | Prometheus and Grafana | Cluster and service metrics dashboards. The Grafana and Prometheus UIs are not installed |
| `--disable-spark` | The Spark operator | Running Spark jobs (`spark` runtime functions) |
| `--disable-mpi` | The MPI operator | Running distributed training jobs with the `mpijob` runtime |
| `--disable-model-monitoring` | Kafka and TimescaleDB | [Model monitoring](../model-monitoring/index.md), which stores its stream data and results in them |

Disabled components are also left out of the access URLs table printed at the end of the installation.

### Ingress

```bash
./scripts/install.py --enable-ingress           # IngressClass "nginx"
./scripts/install.py --enable-ingress traefik   # another class
```

This turns on the chart's Ingresses, with hosts `mlrun.<host>`, `mlrun-api.<host>`, `jupyter.<host>` and `nuclio.<host>`, 
where `<host>` is the external host address. A host or class you set in a `-f` file is kept. 
The installer does **not** install an ingress controller. One providing the class must already run in the cluster 
(check with `kubectl get ingressclass`); otherwise the checks warn and the Ingresses do nothing until one exists. 
The host names must resolve to the controller, through wildcard DNS or a name such as `203.0.113.10.nip.io`. 
With a bare IP the installer prints the `/etc/hosts` line to add.

### Local registry

```bash
./scripts/install.py --local-registry --enable-ingress
```

Runs a `registry:2.8.3` registry in the namespace, backed by a 10Gi PVC on the default StorageClass, and configures MLRun and Nuclio to use it over plain HTTP. 
No registry account is needed. Without `--enable-ingress` the registry is `local-registry.<namespace>.svc.cluster.local:5000`, which the nodes cannot resolve, so function pods fail to pull. 
With ingress, you must also:

1. Add the registry host to the CoreDNS `hosts` block, so pods can push. The installer prints the exact entry. 
   If it cannot find your ingress controller Service, set `INGRESS_CONTROLLER_SERVICE` to its `namespace/name`.
2. Add the registry host to `/etc/hosts` on Docker Desktop, so the node can pull.
3. Add the registry host to `insecure-registries` under **Settings → Docker Engine**, then **Apply & Restart**.

The installer prints these steps but changes none of them itself. 
`--local-registry` with `--enable-ingress` is refused when the external host address is `localhost` or `127.0.0.1`, 
because the registry host would resolve to each pod itself. Use another address, such as `host.docker.internal`.
See [Local registry](https://github.com/mlrun/ce/blob/development/scripts/README.md#local-registry) in the installer README for details.

### OpenTelemetry

```bash
./scripts/install.py --enable-otel collector   # operator + collector: a metrics pipeline
./scripts/install.py --enable-otel             # full: all four pieces
```

The four pieces can also be enabled one by one: `--enable-otel-operator`, `--enable-otel-collector`, `--enable-otel-instrumentation` and `--enable-otel-namespace-label`. 
`--enable-otel off` turns them all off. A mode sets the complete state, so the flags combine left to right. See also [Configure OTel](./mlrun-ce-installation-notes.md#configure-otel).

## Uninstall

```bash
./scripts/install.py uninstall                 # keeps data
./scripts/install.py uninstall --hard-clean    # deletes data
./scripts/install.py uninstall --dry-run       # shows what would be removed
```

`uninstall` removes the Helm release and the objects the installer created outside the chart (the pull secret and the local registry). 
A secret you created yourself stays. It keeps the namespace, the chart's CRDs and every PVC, so a reinstall finds its data.

`--hard-clean` also deletes every PVC in the namespace and the PVs bound to them, the Nuclio functions and API gateways, 
the MLRun run and build pods, and the chart's Kafka resources. **This is irreversible.** 
It lists what it will delete and asks first; `--non-interactive` skips the question. 
The namespace is kept unless you pass `--delete-namespace`.

## Flag reference

Setting a flag's environment variable to `true` or `1` has the same effect as the flag. Any other value, `false` and `0` included, leaves it off. 
An empty variable counts as unset.

| Flag | Variable | Effect |
|---|---|---|
| `-h`, `--help` | | Show help |
| `-v`, `--version` | | Print the installer version (the chart version it ships with) |
| `--uninstall` | `UNINSTALL` | Same as the `uninstall` command |
| `--hard-clean` | `HARD_CLEAN` | With `uninstall`: also delete the namespace's PVCs and PVs, Nuclio functions, MLRun run pods, the chart's Kafka, Nuclio build Jobs and the Spark webhook certificate |
| `--delete-namespace` | `DELETE_NAMESPACE` | With `--hard-clean`: also delete the namespace, without asking |
| `--dry-run` | `DRY_RUN` | Change nothing: render the chart, or list what `uninstall` would remove |
| `--non-interactive` | `NON_INTERACTIVE`, `CI` | Never prompt; exit 1 on a missing required value |
| `--config FILE` | `CONFIG_FILE` | Read defaults from the `installer:` block of a `ce-config.yaml` |
| `-f`, `--values FILE` | | Base Helm values file |
| `--chart-path DIR` | `CHART_PATH` | Install a local chart directory; not with `--ce-version` |
| `--update-dependencies` | `UPDATE_DEPENDENCIES` | With `--chart-path`: fetch missing chart dependencies |
| `--ce-version VERSION` | `CE_VERSION` | Install this published chart version (default: the newest stable one); not with `--chart-path` |
| `--devel` | `DEVEL` | Without `--ce-version`: the newest chart, prereleases included |
| `--skip-secret` | `SKIP_REGISTRY_SECRET` | Use an existing pull secret instead of creating one |
| `--skip-validators` | `SKIP_VALIDATORS` | Skip the pre-install checks |
| `--local-registry` | `LOCAL_REGISTRY` | Run a registry in the cluster |
| `--kube-context` | `KUBE_CONTEXT` | kubeconfig context for kubectl and helm (default: the current one) |
| `--enable-ingress [CLASS]` | `ENABLE_INGRESS`, `INGRESS_CLASS` | Turn on the chart's Ingresses (class `nginx` by default) |
| `--enable-otel [MODE]` | | `off`, `collector` or `full` (the default for a bare flag) |
| `--enable-otel-operator` | `ENABLE_OTEL_OPERATOR` | Install the OpenTelemetry operator |
| `--enable-otel-collector` | `ENABLE_OTEL_COLLECTOR` | Deploy the OpenTelemetry Collector |
| `--enable-otel-instrumentation` | `ENABLE_OTEL_INSTRUMENTATION` | Create the Instrumentation resource |
| `--enable-otel-namespace-label` | `ENABLE_OTEL_NAMESPACE_LABEL` | Auto-instrument every Python pod in the namespace |
| `--disable-system-monitoring` | `DISABLE_SYSTEM_MONITORING` | Leave out Prometheus and Grafana |
| `--disable-spark` | `DISABLE_SPARK` | Leave out the Spark operator |
| `--disable-mpi` | `DISABLE_MPI` | Leave out the MPI operator |
| `--disable-model-monitoring` | `DISABLE_MODEL_MONITORING` | Leave out Kafka and TimescaleDB |
| `--show-progress` | `SHOW_PROGRESS` | Show a live table of rollouts while Helm runs (terminals only) |
| `--log-level LEVEL` | `LOG_LEVEL` | `info` (the default) or `warning`: only warnings, errors, prompts and the result reach the terminal; the debug log still has everything |
| `--trigger-migrations` | `TRIGGER_MIGRATIONS` | On an upgrade, run the MLRun DB migrations for this run only |

Variables without a flag:

| Variable | Default | Meaning |
|---|---|---|
| `NAMESPACE` | `mlrun` | Target namespace |
| `RELEASE_NAME` | `mlrun-ce` | Helm release name |
| `REGISTRY_USERNAME` | | Registry username |
| `REGISTRY_PASSWORD` | | Registry password |
| `REGISTRY_PASSWORD_FILE` | | File holding only the password; `REGISTRY_PASSWORD` wins if both are set |
| `REGISTRY_SERVER` | `https://index.docker.io/v1/` | Registry server for the pull secret |
| `REGISTRY_EMAIL` | | Registry email (optional) |
| `REGISTRY_URL` | | Where images are pushed, for example `index.docker.io/myuser` |
| `REGISTRY_SECRET_NAME` | `registry-credentials` | Pull secret name |
| `EXTERNAL_HOST_ADDRESS` | suggested | The address you use to reach the cluster's NodePorts, and the base of the ingress hosts |
| `INGRESS_CONTROLLER_SERVICE` | | `namespace/name` of the ingress controller Service, for the local registry's CoreDNS entry |
| `MLRUN_VERSION` | the chart's | MLRun API and UI image tag |
| `NUCLIO_VERSION` | the chart's | Nuclio controller and dashboard image tag |
| `HELM_REPO_URL` | `https://mlrun.github.io/ce` | Chart repository |
| `HELM_TIMEOUT` | `960s` | Helm's wait timeout for install, upgrade and uninstall |
| `MIN_K8S_VERSION` | | Also warn below this Kubernetes version (the 1.34 warning always applies) |
| `PROGRESS_INTERVAL_SEC` | `10` | Refresh interval of `--show-progress` |
| `GENERATED_VALUES_FILE` | `mlrun-ce-values.yaml` | Where the installer writes its values; rewritten every run. An upgrade's `.previous.yaml` and `.changes.yaml` files go beside it |
| `INSTALL_LOG_FILE` | `mlrun-ce-install.log` | The debug log; rewritten every run |
