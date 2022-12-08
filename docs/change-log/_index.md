(change-log)=
# Change log
- [v1.2.0](#v1-2-0)
- [Open issues](#open-issues)
- [Limitations](#limitations)
- [Deprecations](#deprecations)


## v1.2.0

### New and updated features

#### Artifacts
- Support for artifact tagging:
    SDK: Add `tag_artifacts`  and `delete_artifacts_tags` that can be used to modify existing artifacts tags and have more than one version for an artifact.
    API: Introduce new endpoints in `/projects/<project>/tags`.
	UI: You can add and edit artifact tags in the UI.
    
#### Auth
- Support S3 profile and assume-role when using `fsspec`.
- Support GitHub fine grained tokens.

#### Documentation
- Restructured, and new content


#### Feature store
- Support Redis as an online feature set. (See [Redis target store](../data-prep/ingest-data-fs.html#redis-target-store-tech-preview).)
- Support GCP objects as a data source for the feature store.
- Fully support ingesting with pandas engine, now equivalent to ingestion with `storey` engine:
   - Support DataFrame with multi-index.
   - Support mlrun steps when using pandas engine: `OneHotEncoder` , `DateExtractor`, `MapValue`, `Imputer` and `FeatureValidation`.
- Add new step: `DropFeature` for pandas and storey engines.
- Add param query for `get_offline_feature` for filtering the output.

#### Frameworks
- Add `HuggingFaceModelServer` to `mlrun.frameworks` at `mlrun.frameworks.huggingface` to serve `HuggingFace` models.

#### Functions
- Add `function.with_annotations({"framework":"tensorflow"})` to user created functions.
- Add `overwrite_build_params` to `project.build_function()` so the user can choose whether or not to keep the build params that were used in previous function builds.
- `deploy_function` has a new option of mock deployment that allows running the function locally


#### Installation
- Add option to install `google-cloud` requirements using `mlrun[google-cloud]`:  when installing MLRun for integration with GCP clients, only compatible packages are installed.


#### Models
- The Labels in the **Models > Overview** tab can be edited

#### Third party integrations
- Supports Confluent Kafka (Tech Preview)

#### Internal
- Refactor artifacts endpoints to follow the MLRun convention of `/projects/<project>/artifacts/...`. (The previous API will be deprecated in a future release.)
- Add `/api/_internal/memory-reports/` endpoints for memory related metrics to better understand the memory consumption of the API.
- Improve the HTTP retry mechanism.
- Support a new lightweight mechanism for KFP pods to pull the run state they triggered. Default behavior is legacy, which pulls the logs of the run to figure out the run state. 
The new behavior can be enabled using a feature flag configured in the API.

### Breaking changes

- Feature store: Ingestion using pandas now takes the dataframe and creates an index out of the entity column (and removes it from 
being a column in this df). This could cause breakage for existing custom steps when using a pandas engine.

### Closed issues

- Support logging artifacts larger than 5GB to V3IO. [View in Git](https://github.com/mlrun/mlrun/issues/2455).
- Limit KFP to kfp~=1.8.0, <1.8.14 due to non-backwards changes done in 1.8.14 for ParallelFor, which isn’t compatible with the MLRun managed KFP server (1.8.1) .[View in Git](https://github.com/mlrun/mlrun/issues/2516).
- Add `artifact_path` enrichment from project `artifact_path`. Previously, the parameter wasn't applied to project runs when defining `project.artifact_path`. [View in Git](https://github.com/mlrun/mlrun/issues/2507).
- Align timeouts for requests that are getting re-routed from worker to chief (for projects/background related endpoints). [View in Git](https://github.com/mlrun/mlrun/issues/2565).
- Fix legacy artifacts load when loading a project. Fixed corner cases when legacy artifacts were saved to yaml and loaded back into the system using `load_project()`. [View in Git](https://github.com/mlrun/mlrun/issues/2584).
- Fix artifact `latest` tag enrichment to happen also when user defined a specific tag. [View in Git](https://github.com/mlrun/mlrun/issues/2572).
- Fix zip source extraction during function build. [View in Git](https://github.com/mlrun/mlrun/issues/2588).
- Fix Docker compose deployment so Nuclio is configured properly with a platformConfig file that sets proper mounts and network configuration for Nuclio functions, meaning that they run in the same network as MLRun. [View in Git](https://github.com/mlrun/mlrun/issues/2601).
- Workaround for background tasks getting cancelled prematurely, due to the current FastAPI version that has a bug in the starlette package it uses. The bug caused the task to get cancelled if the client’s http connection was closed before the task was done. [View in Git](https://github.com/mlrun/mlrun/issues/2618).
- Fix run fails after deploying function without defined image. [View in Git](https://github.com/mlrun/mlrun/pull/2530).
- Scheduled jobs failed on GKE with resource quota error. [View in Git](https://github.com/mlrun/mlrun/pull/2520).
- Can now delete a model via tag. [View in Git](https://github.com/mlrun/mlrun/pull/2433).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.2.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.2.0)


## Open issues

| ID   | Description                                            | Workaround                                    | Opened |
| ---- | -------------------------------------------------------| --------------------------------------------- | ------ |
| 2849 | Cannot pickle a class inside an mlrun function         | Use cloudpickle instead of pickle             | 1.2.0  |
| 2516 | Feature store is not backwards compatible              | Planned for a future release                  | 1.2.0  |
| 2223 | Cannot deploy a function when notebook names contain "." (ModuleNotFoundError) | Do not use "." in notebook name | 1.0.0  |
| 2199 | Spark operator job fails with default requests args       | NA                                         | 1.0.0 |
| 1584 | Cannot run `code_to_function` when filename contains special characters | Do not use special characters in filenames | 1.0.0 |
| 2637 | Running a workflow whose project has `init_git=True`, results in Project error |Run `git config --global --add safe.directory '*'` (can substitute specific directory for *).                             | 1.1.0 |

## Limitations


| ID   | Description                                                    | Workaround                           | Opened | 
| ---- | -------------------------------------------------------------- | ------------------------------------ | ----------|      
| 2014 | Model deployment returns ResourceNotFoundException (Nuclio error that Service <name> is invalid.) | Verify that all `metadata.labels` values are 63 characters or less (Kubernetes limitation). |  v1.0.0  |
 

## Deprecations

    
| In v.  | ID |Description                                                          |
|------ | ---- | --------------------------------------------------------------------|
| 1.0.0 |     | MLRun / Nuclio do not support python 3.6                             |