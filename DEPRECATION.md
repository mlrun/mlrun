# Deprecation Guidelines

This document describes the **deprecation procedure** in MLRun.  
Follow these steps when deprecating a parameter, method, class, endpoint, or query parameter.  
Use the checklist at the end to verify all relevant updates are applied.

---

## General Rules

- **Backward compatibility** is kept for **2 minor versions**.  
  Example: if a parameter is deprecated in `1.10.0`, it is removed in `1.12.0`.
- Always specify what should be used instead.  
  If there is no replacement, explain why.
- Every MLRun version should have a Jira ticket to gather all deprecations and removals.  
  Example: *ML-9365: 1.8.0 Deprecations and removals*

---

## Process

### 1. Planning phase
- Product team prepares a list of items to be deprecated in the upcoming version.

### 2. Development kickoff
- Developers remove all deprecations from **2 minor versions ago** at the start of the release cycle.
- Track removals using:
  - Code comments (`# TODO: Remove in x.y.z`)
  - Matching Jira ticket
  - Docs changelog
- Communicate removals to the **Customer Success** team (via Jira).

### 3. During development cycle
- Developers coordinate new deprecations with the Jira ticket.
- Update the ticket for **every** change.

---

## Special Cases

### 1. Removing without warning
Accepted only if:
- Backward compatibility break is agreed upon and approved with Customer Success.
- Documented in the Jira ticket under a dedicated section with explanation.

### 2. Breaking upgrade from old versions
Sometimes legacy code is required for migrations (e.g. migrating artifacts).  
Removing such code requires:
- Agreement on breaking upgrade compatibility.
- Documentation in the Jira ticket under a dedicated section with explanation.

---

## How to Deprecate

### 1. Parameter
```python
if uid:
    warnings.warn(
        "'uid' is deprecated in 1.10.0 and will be removed in 1.12.0, use 'tree' instead.",
        # TODO: Remove this in 1.12.0
        FutureWarning,
    )
```

### 2. Method

```python
# TODO: remove in 1.12.0
@deprecated(
    version="1.10.0",
    reason="'verify_base_image' will be removed in 1.10.0, use 'prepare_image_for_deploy' instead",
    category=FutureWarning,
)
def verify_base_image(self):
    pass
```

### 3. Class

```python
# TODO: Remove in 1.12.0
@deprecated(
    version="1.10.0",
    reason="v1alpha1 mpi will be removed in 1.12.0, use v1 instead",
    category=FutureWarning,
)
class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
    pass
```

### 4. Endpoint

```python
# TODO: Remove in 1.12.0
@router.get(
    "/runs",
    deprecated=True,
    description="/runs is deprecated in 1.10.0 and will be removed in 1.12.0. "
    "Use /projects/{project}/runs/ instead",
)
async def list_runs():
    pass
```

API changes are not documented in MLRun docs.
Deprecation is only visible in Swagger and required when the SDK uses the endpoint.

### 5. Query Parameter
```python
limit: int = Query(
    None,
    deprecated=True,
    description="'limit' query param is deprecated in 1.10.0 and will be removed in 1.12.0. "
    "Use page and page_size instead.",
)
```
API changes are not documented in mlrun docs. They are needed when the SDK may be using the endpoint. 
The deprecation warning is only visible in Swagger.

---

## Checklist
- **Update “Deprecations and removals” Jira ticket**  
  Link the PR in the ticket.

- **Update MLRun docs**  
  Ensure the changelog reflects the deprecation or removal.

- **Update repositories affected by the deprecation/removal:**
  - [mlrun/functions](https://github.com/mlrun/functions)
  - [mlrun/marketplace](https://github.com/mlrun/marketplace)
  - [mlrun/demos](https://github.com/mlrun/demos)
  - [mlrun/test-notebooks](https://github.com/mlrun/test-notebooks)
  - [mlrun/examples](https://github.com/mlrun/mlrun/tree/development/examples)
