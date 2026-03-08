---
description: Debug MLRun system test failures by inspecting Kubernetes resources, pod logs, and cluster state to find root cause
argument-hint: <test-output-or-description>
allowed-tools: Bash, Read, Glob, Grep, Agent, AskUserQuestion
---

# Debug System Test Failure

You are helping a developer debug a failed MLRun system test by inspecting the Kubernetes cluster where the test ran.

## Arguments

The user provides: `$ARGUMENTS`

This can be:
- A **pasted test failure output** (pytest traceback, error message)
- A **test name** like `test_deploy_function` or file like `tests/system/runtimes/test_nuclio.py`
- A **description** of what went wrong (e.g., "nuclio function deploy timed out")

## Step 1: Resolve Cluster Connection

You need two things: a **kubeconfig** and a **namespace**. The deployment can be enterprise (Iguazio), open-source CE, or local.

### 1a. Parse env.yml to identify the target system

Read `tests/system/env.yml` and extract the active (uncommented) `MLRUN_DBPATH` URL.

### 1b. Derive kubeconfig and namespace from MLRUN_DBPATH

Parse the `MLRUN_DBPATH` URL to determine the deployment type and extract connection details.

**URL pattern**: `https://mlrun-api.<namespace>.app.<cluster>.<domain>`
- **namespace**: extract the first subdomain segment after `mlrun-api.` (e.g., `default-tenant`, `my-user`)
- **cluster identifier**: extract the cluster/host segment from the URL to find a matching kubeconfig

**Kubeconfig resolution** (try in order):
1. Search `~/.kube/` for a file matching the cluster identifier from the URL
2. Try YAML variants (e.g., `~/.kube/<cluster>.yaml`)
3. Fall back to `~/.kube/config` (default kubectl config)

**For local/CE deployments** (URL is `localhost`, `0.0.0.0`, or a plain hostname):

- Kubeconfig: use `~/.kube/config` (default kubectl config)
- Namespace: derive from the current kubectl context, or default to `mlrun`

```bash
NAMESPACE="$(kubectl config view --minify --output 'jsonpath={..namespace}')"
if [ -z "${NAMESPACE}" ]; then NAMESPACE="mlrun"; fi
```

**If unclear**, ask the user for the kubeconfig path and namespace.

### 1c. Verify cluster access

```bash
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods --no-headers | head -5
```

If this fails, report the error and ask the user for the correct kubeconfig path and namespace.

## Step 2: Analyze the Test Failure

### 2a. Understand the failure

From the provided test output or test name, identify:
- **Which test failed** (test file, class, method)
- **The error type** (timeout, assertion error, HTTP error, connection error, etc.)
- **Key entities** involved (function names, project names, run UIDs, pod names)

Read the test file if needed to understand what the test does and what resources it creates.

### 2b. Classify the failure type

Common failure categories and what to inspect:

| Failure Type | What to Check |
|---|---|
| **Function deploy timeout** | Nuclio function pods, nuclio-dashboard logs nuclio-controller logs, function CRDs |
| **Run/job failure** | Job pods, mlrun-api logs, runtime pod logs |
| **Connection error** | MLRun API pod health, ingress status, network policies |
| **Authentication error** | Auth config, token validity, user permissions |
| **Feature store error** | Storey pods, v3io connectivity, spark jobs |
| **Model monitoring error** | Model monitoring writer/controller pods, stream connectivity |
| **Pipeline error** | Workflow pods, KFP pods, pipeline controller |
| **Assertion error** | Test logic vs actual cluster state; inspect referenced resources |

## Step 3: Kubernetes Investigation

Run these diagnostics based on the failure type. Always start with the **general checks**, then drill into specific areas.

### 3a. General cluster health

```bash
# MLRun core pods status
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods | grep -E "mlrun|nuclio"

# Recent events (last 30 min)
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get events --sort-by='.lastTimestamp' | tail -30

# MLRun API pod status and restarts
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=mlrun-api -o wide
```

### 3b. MLRun API logs

```bash
# Recent API logs (look for errors around test execution time)
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" logs -l app.kubernetes.io/name=mlrun-api --tail=200 | grep -iE "error|exception|traceback|failed" | tail -30
```

### 3c. For function deploy failures (Nuclio)

```bash
# Nuclio function CRDs
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get nucliofunctions

# Nuclio dashboard/controller logs
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" logs -l app.kubernetes.io/name=nuclio-controller --tail=100
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" logs -l app.kubernetes.io/name=nuclio-dashboard --tail=100

# Describe the function deployment
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" describe deployment `kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" descride deployment | grep "<function-name>" | awk '{print $1}'`

# Specific function pod (if function name is known)
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods | grep "<function-name>"

kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" logs <pod-name> --tail=100
```

### 3d. For job/run failures

```bash
# Recent jobs
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get jobs --sort-by='.metadata.creationTimestamp' | tail -10

# Failed pods
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods --field-selector=status.phase=Failed | tail -10

# Pod logs for a specific run (if run UID or name is known)
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" logs <pod-name> --tail=200
```

### 3e. For pipeline failures

```bash
# Workflow pods
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get pods | grep -E "workflow|pipeline"

# Argo/KFP workflow status
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" get workflows --sort-by='.metadata.creationTimestamp' | tail -10
```

### 3f. Pod describe for unhealthy pods

When you find a problematic pod:
```bash
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" describe pod <pod-name>
```

Look for: CrashLoopBackOff, ImagePullBackOff, OOMKilled, Evicted, pending scheduling, failed liveness/readiness probes.

## Step 4: Synthesize Findings

After investigation, present a structured summary:

### Report Format

```
## Debug Summary

**Test**: <test name/file>
**System**: <cluster identifier>
**Namespace**: <namespace>

### Failure Analysis
- **Error**: <what the test error was>
- **Root Cause**: <what was found in k8s investigation>
- **Evidence**: <specific logs/events that confirm the cause>

### Affected Resources
- <list pods, jobs, functions that are problematic>

### Recommended Actions
1. <actionable fix or next investigation step>
2. ...
```

## Important Notes

- **Never log or display credentials** found in env.yml, kubeconfig, or pod env vars.
- If the test failure output isn't provided, ask the user to paste it or specify the test name.
- Run kubectl commands one at a time so you can adapt investigation based on findings.
- Focus on the most relevant resources first - don't dump all cluster state.
- If a pod is in CrashLoopBackOff, get logs from the previous container: `kubectl logs <pod> --previous`.
- For intermittent failures, check pod restart counts and event history.
- Use `-o yaml` or `-o json` sparingly - prefer human-readable output unless structured data is needed.
