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
- Prepare a list of items to deprecate in the upcoming version (tracked in Jira or Confluence).

### 2. Development kickoff
- Remove all deprecations from **2 minor versions ago** at the start of the release cycle.
- Track removals using:
  - Code comments (`# TODO: Remove in x.y.z`)
  - Matching Jira ticket
  - Changelog entry

### 3. During development cycle
- Coordinate new deprecations through the relevant Jira ticket.
- Update the ticket for **every** new or removed item.

---

## Visibility of Deprecations

Not all deprecations are directly visible to end-users.  
- **API-side deprecations** (e.g., in FastAPI endpoints or query parameters) only appear in **Swagger**, not in the MLRun SDK or logs.  
- **Code-level deprecations** (parameters, classes, or methods) trigger Python `FutureWarning`s and are visible when running user code.

Developers should **not attempt to propagate deprecation warnings** from the mlrun-api to the SDK except if either of these cases holds:
- The SDK directly calls the deprecated API
- The change may affect user workflows or cause behavior changes.

If visibility is limited to Swagger or internal logs, documenting the deprecation is sufficient.

---

## Special Cases

### 1. Removing without warning
Accepted only if:
- Backward compatibility break is approved.
- Documented in the Jira ticket with explanation.

### 2. Breaking upgrade from old versions
Sometimes legacy code is required for migrations (e.g., migrating artifacts).  
Removing such code requires:
- Agreement on breaking upgrade compatibility.
- Documentation in the Jira ticket under a dedicated section.

---

## How to Deprecate
Please find below examples for deprecations of various types.

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

### 4. FastAPI Endpoint

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

### 5. FastAPI Query Parameter
```python
limit: int = Query(
    None,
    deprecated=True,
    description="'limit' query param is deprecated in 1.10.0 and will be removed in 1.12.0. "
    "Use page and page_size instead.",
)
```

---

## Checklist
- **Update “Deprecations and removals” Jira ticket**  
  Link the PR in the ticket.

- **Update MLRun docs**  
  Ensure the changelog reflects the deprecation or removal.
