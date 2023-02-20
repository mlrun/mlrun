(change-log)=
# Change log
- [v1.3.0](#v1.3.0)
- [v1.2.1](#v1-2-1)
- [v1.2.0](#v1-2-0)
- [v1.1.3](#1-1-3)
- [v1.0.6](#v1-0-6)
- [v1.0.5](#v1-0-5)
- [v1.0.4](#v1-0-4)
- [v1.0.3](#v1-0-3)
- [v1.0.2](#v1-0-2)
- [v1.0.0](#v1-0-0)
- [Open issues](#open-issues)
- [Limitations](#limitations)
- [Deprecations](#deprecations)

## v1.3.0

### Client/server matrix



- Base images mlrun/mlrun:1.3.0 etc. are based on python 3.9.16. <br>
      For Iguazio <=3.5.2:
	  
	1. Configure the Jupyter service with the env variable`JUPYTER_PREFER_ENV_PATH=false`.
    2. Within the Jupyter service, open a terminal and run:
     
	```
	 conda create -n python39 python=3.9 ipykernel -y
      conda activate python39
      ./align_mlrun.sh
	```
	  
- v1.3.0 retains support for mlrun base images that are based on python 3.7. To differentiate between the images, the images based on python 3.7 have the suffix: `-py37`.

### New and updated features


#### APIs

- New APIs may require a new version of the MLRun client/server **ML-3266**

**Modified APIs**
- These APIs now only return reasons in kwargs: `log_and_raise`, `generic_error_handler`, `http_status_error_handler`.
 
- See also **[Deprecated APIs](#api-130)**.
 
 
#### Infrastructure

- MLRun supports Python 3.9.16

#### Feature store

- Spark engine supports the steps: MapValues, Imputer, OneHotEncoder, DropFeatures; and supports extracting the time parts from the date in the DateExtractor step.
- Supports creating a feature vector over several feature sets with different entity. See [Creating an offline feature vector](../feature-store/feature-vectors.html#creating-an-offline-feature-vector)
- Supports SQLSource for batch ingestion and real time ingestion in the feature store, and SQLTarget (supports storey, does not support Spark). 
   See [SQL data source](../data-prep/ingest-data-fs.html#sql-data-source) and [SQL target store](../data-prep/ingest-data-fs.html#sql-target-store)
- The RedisNoSqlTarget now supports the Spark engine.
- The username and password for the RedisNoSqlTarget aare now configured using secrets, as <prefix_>REDIS_USER <prefix_>REDIS_PASSWORD where <prefix> is optional 
   RedisNoSqlTarget 'credentials_prefix' parameter. See [Redis target store](../data-prep/ingest-data-fs.html#redis-target-store)
- Offline data can be registered as feature sets. See [Create a feature set without ingesting its data](../feature-store/feature-sets#create-a-feature-set-without-ingesting-its-data).

#### Projects

- When defining a new project from scratch, there is now a default` context` directory, by default, "./", which is the directory that the MLRun client runs from.  

#### Serving graphs

- ML-2506 Allow providing a list of steps for the "after" argument in the `add_step()` method
- ML-1167 Add support for graphs that split and merge (DAG)
- ML-2507 Supports configuring of consumer group name for steps following QueueSteps

#### Storey
- The event time in storey events is now taken from the `timestamp_key`. If the `timestamp_key` is not defined for the event, then the time is taken from the processing-time metadata. [View in Git](https://github.com/mlrun/storey/pull/394).


#### Third party integrations
- Supports Confluent Kafka as a feature store data-source. See [Confluent Kafka data source](../data-prep/ingest-data-fs.html#confluent-kafka-data-source) (Tech Preview).

#### UI
- The new **Projects** home page provides easy and intuitive access to the common project tasks.


<a id="api-130"></a>
### Deprecated and removed APIs
Starting with v1.3.0, and continuing in subsequent releases, obsolete functions are getting removed from the code.

**Deprecated and removed from v1.3.0 code**<br>
The following MLRun APIs have been deprecated since at least v1.0.0. Until now, a warning appeared if you attempted to use theצ. 
They are now removed from the code:
- project.workflows`
- project.functions`
- project.artifacts`
- pod_status header` from response to get_log REST API
- client-spec` from response to health API 
- submit_pipeline_legacy` REST API
- get_pipeline_legacy` REST API
- Five runtime legacy REST APIs, such as: `list_runtime_resources_legacy`, `delete_runtime_object_legacy` etc.
- `project.func()` (instead, use `project.get_function()`)
- `project.create_vault_secrets()` (instead, use `project.set_secrets()`)
- `project.get_vault_secret()`
- `MlrunProjectLegacy` class
- Feature-store: usage of state in graph (replaced by step). For example, in targets.py: add_writer_state, and the after_state parameter in _init_ methods.
- httpdb runtime-related APIs using the deprecated runtimes REST APIs: `delete_runtime_object` etc.
- `mount_path` parameter in mount_v3io(). Instead, use the `volume_mounts` parameter
- Dask properties: `gpus`, `with_limits`, `with_requests`
- `NewTask`

**Deprecated, will be removed in v1.5.0**<br>
These APIs will be removed from the v1.5.0 code. A FutureWarning appears if you try to use them in v1.3.0:
- project-related parameters of `set_environment`. 
- `AbstractSparkJobSpec.gpus`, `KubeResource.gpus`, `DaskCluster.gpus`. Use the `with_limits` method instead.
- `mount_v3io_legacy` (mount_v3io no longer calls it)
- `mount_v3io_extended `
- class `LegacyArtifact`
- `init_functions` in pipelines 
- The entire `mlrun/mlutils` library

### Closed issues
- Can now pickle a class inside an mlrun function. [View in Git](https://github.com/mlrun/mlrun/pull/
- Fix: Project page was displaying an empty list after an upgrade [View in Git](https://github.com/mlrun/ui/pull/1611)
- Jobs and Workflows pages now display the tag of the executed job as defined in the API **ML-2534**
- Fix: Ingestion with add_aggregation over spark, with aggregation operation 'sqr' or 'stdvar'. Previously failed with `AttributeError: module 'pyspark.sql.functions' has no attribute 'stdvar'/'sqr'`. [View in Git](https://github.com/mlrun/mlrun/pull/3062).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.3.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.3.0)





## v1.2.1

### New and updated features

#### Feature store
- Supports ingesting Avro-encoded Kafka records. [View in Git](https://github.com/mlrun/mlrun/issues/2649).

### Closed issues

- Fix: the **Projects | Jobs | Monitor Workflows** view is now accurate when filtering for > 1 hour. [View in Git](https://github.com/mlrun/mlrun/pull/2786).
- The Kubernetes **Pods** tab in **Monitor Workflows** now shows the complete pod details. [View in Git](https://github.com/mlrun/mlrun/pull/1576).
- Update the tooltips in **Projects | Jobs | Schedule** to explain that day 0 (for cron jobs) is Monday, and not Sunday. 
[View in Git](https://github.com/mlrun/ui/pull/1571).
- Fix UI crash when selecting **All** in the **Tag** dropdown list of the **Projects | Feature Store | Feature Vectors** tab. [View in Git](https://github.com/mlrun/ui/pull/1549).
- Fix: now updates `next_run_time` when skipping scheduling due to concurrent runs. [View in Git](https://github.com/mlrun/mlrun/pull/2862).
- When creating a project, the error `NotImplementedError` was updated to explain that MLRun does not have a 
DB to connect to. [View in Git](https://github.com/mlrun/mlrun/pull/2856).
- When previewing a **DirArtifact** in the UI, it now returns the requested directory. Previously it was returning the directory list from the root of the container. [View in Git](https://github.com/mlrun/mlrun/pull/2592).
- Load source at runtime or build time now fully supports .zip files, which were not fully supported previously.

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.2.1)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.2.1)


## v1.2.0

### New and updated features

#### Artifacts
- Support for artifact tagging:
   - SDK: Add `tag_artifacts`  and `delete_artifacts_tags` that can be used to modify existing artifacts tags and have 
    more than one version for an artifact.
    - UI: You can add and edit artifact tags in the UI.
    - API: Introduce new endpoints in `/projects/<project>/tags`.
    
#### Auth
- Support S3 profile and assume-role when using `fsspec`.
- Support GitHub fine grained tokens.

#### Documentation
- Restructured, and added new content.

#### Feature store
- Support Redis as an online feature set for storey engine only. (See [Redis target store](../data-prep/ingest-data-fs.html#redis-target-store).)
- Fully supports ingesting with pandas engine, now equivalent to ingestion with `storey` engine (TechPreview):
   - Support DataFrame with multi-index.
   - Support mlrun steps when using pandas engine: `OneHotEncoder` , `DateExtractor`, `MapValue`, `Imputer` and `FeatureValidation`.
- Add new step: `DropFeature` for pandas and storey engines. (TechPreview)
- Add param query for `get_offline_feature` for filtering the output.

#### Frameworks
- Add `HuggingFaceModelServer` to `mlrun.frameworks` at `mlrun.frameworks.huggingface` to serve `HuggingFace` models.

#### Functions
- Add `function.with_annotations({"framework":"tensorflow"})` to user-created functions.
- Add `overwrite_build_params` to `project.build_function()` so the user can choose whether or not to keep the 
build params that were used in previous function builds.
- `deploy_function` has a new option of mock deployment that allows running the function locally.

#### Installation
- New option to install `google-cloud` requirements using `mlrun[google-cloud]`:  when installing MLRun for integration 
with GCP clients, only compatible packages are installed.

#### Models
- The Labels in the **Models > Overview** tab can be edited.



#### Internal
- Refactor artifacts endpoints to follow the MLRun convention of `/projects/<project>/artifacts/...`. (The previous API will be deprecated in a future release.)
- Add `/api/_internal/memory-reports/` endpoints for memory related metrics to better understand the memory consumption of the API.
- Improve the HTTP retry mechanism.
- Support a new lightweight mechanism for KFP pods to pull the run state they triggered. Default behavior is legacy, 
which pulls the logs of the run to figure out the run state. 
The new behavior can be enabled using a feature flag configured in the API.

### Breaking changes

- Feature store: Ingestion using pandas now takes the dataframe and creates indices out of the entity column 
(and removes it as a column in this df). This could cause breakage for existing custom steps when using a pandas engine.

### Closed issues

- Support logging artifacts larger than 5GB to V3IO. [View in Git](https://github.com/mlrun/mlrun/issues/2455).
- Limit KFP to kfp~=1.8.0, <1.8.14 due to non-backwards changes done in 1.8.14 for ParallelFor, which isn’t compatible with the MLRun managed KFP server (1.8.1). [View in Git](https://github.com/mlrun/mlrun/issues/2516).
- Add `artifact_path` enrichment from project `artifact_path`. Previously, the parameter wasn't applied to project runs when defining `project.artifact_path`. [View in Git](https://github.com/mlrun/mlrun/issues/2507).
- Align timeouts for requests that are getting re-routed from worker to chief (for projects/background related endpoints). [View in Git](https://github.com/mlrun/mlrun/issues/2565).
- Fix legacy artifacts load when loading a project. Fixed corner cases when legacy artifacts were saved to yaml and loaded back into the system using `load_project()`. [View in Git](https://github.com/mlrun/mlrun/issues/2584).
- Fix artifact `latest` tag enrichment to happen also when user defined a specific tag. [View in Git](https://github.com/mlrun/mlrun/issues/2572).
- Fix zip source extraction during function build. [View in Git](https://github.com/mlrun/mlrun/issues/2588).
- Fix Docker compose deployment so Nuclio is configured properly with a platformConfig file that sets proper mounts and network 
configuration for Nuclio functions, meaning that they run in the same network as MLRun. 
[View in Git](https://github.com/mlrun/mlrun/issues/2601).
- Workaround for background tasks getting cancelled prematurely, due to the current FastAPI version that 
has a bug in the starlette package it uses. The bug caused the task to get cancelled if the client’s HTTP connection 
was closed before the task was done. [View in Git](https://github.com/mlrun/mlrun/issues/2618).
- Fix run fails after deploying function without defined image. [View in Git](https://github.com/mlrun/mlrun/pull/2530).
- Fix scheduled jobs failed on GKE with resource quota error. [View in Git](https://github.com/mlrun/mlrun/pull/2520).
- Can now delete a model via tag. [View in Git](https://github.com/mlrun/mlrun/pull/2433).


### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.2.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.2.0)



## v1.1.3

### Closed issues

- The CLI supports overwriting the schedule when creating scheduling workflow. [View in Git](https://github.com/mlrun/mlrun/pull/2651).
- Slack now notifies when a project fails in `load_and_run()`. [View in Git](https://github.com/mlrun/mlrun/pull/2794).
- Timeout is executed properly when running a pipeline in CLI. [View in Git]https://github.com/mlrun/mlrun/pull/2635).
- Uvicorn Keep Alive Timeout (`http_connection_timeout_keep_alive`) is now configurable, with default=11. This maintains 
API-client connections. [View in Git](https://github.com/mlrun/mlrun/pull/2613).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.3)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.3)

## v1.1.2

### New and updated features

**V3IO**
- v3io-py bumped to 0.5.19.
- v3io-fs bumped to 0.1.15.

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.2)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.2-rc3)

## v1.1.1

### New and updated features

#### API
- Supports workflow scheduling.

#### UI
- Projects: Supports editing model labels.

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.1)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.1)


## v1.1.0

### New and updated features

#### API
-  MLRun scalability: Workers are used to handle the connection to the MLRun database and can be increased to 
improve handling of high workloads against the MLRun DB. You can configure the number of workers for an MLRun 
service, which is applied to the service's user-created pods. The default is 2. 
   - v1.1.0 cannot run on top of 3.0.x.
   - For Iguazio <v3.5.0 number of workers set to 1 by default. To change this number, contact support (helm-chart change required).
   - Multi-instance is not supported for MLrun running on SQLite.
-  Supports pipeline scheduling.
      
#### Documentation
- Added Azure and S3 examples to {ref}`ingest-features-spark`.

#### Feature store
- Supports S3, Azure, GCS targets when using Spark as an engine for the feature store.
- Snowflake as datasource has a connector ID: `iguazio_platform`.
- You can add a time-based filter condition when running `get_offline_feature` with a given vector. 

#### Storey
- MLRun can write to parquet with flexible schema per batch for ParquetTarget: useful for inconsistent or unknown schema.

#### UI

- The **Projects** home page now has three tiles, Data, Jobs and Workflows, Deployment, that guide you through key 
capabilities of Iguazio, and provide quick access to common tasks.
- The **Projects | Jobs | Monitor Jobs** tab now displays the Spark UI URL.
- The information of the Drift Analysis tab is now displayed in the Model Overview.
- If there is an error, the error messages are now displayed in the **Projects | Jobs | Monitor** jobs tab.

#### Workflows
- The steps in **Workflows** are color-coded to identify their status: blue=running; green=completed; red=error.

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.0)

## v1.0.6

### Closed issues
- Import from mlrun fails with "ImportError: cannot import name dataclass_transform".
   Workaround for previous releases:
   Install `pip install pydantic==1.9.2` after `align_mlrun.sh`.
- MLRun FeatureSet was not not enriching with security context when running from the UI. [View in Git](https://github.com/mlrun/mlrun/pull/2250).
- MlRun Accesskey presents as cleartext in the mlrun yaml, when the mlrun function is created by feature set 
   request from the UI. [View in Git](https://github.com/mlrun/mlrun/pull/2250).
   
### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.6)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.6)

## v1.0.5

### Closed issues
- MLRun: remove root permissions. [View in Git](https://github.com/mlrun/mlrun/pull/).
- Users running a pipeline via CLI project run (watch=true) can now set the timeout (previously was 1 hour). [View in Git](https://github.com/mlrun/mlrun/pull/).
- MLRun: Supports pushing images to ECR. [View in Git](https://github.com/mlrun/mlrun/pull/).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.5)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.5)

## v1.0.4

### New and updated features
- Bump storey to 1.0.6.
- Add typing-extensions explictly.
- Add vulnerability check to CI and fix vulnerabilities.

### Closed issues
- Limit Azure transitive dependency to avoid new bug. [View in Git](https://github.com/mlrun/mlrun/pull/2034).
- Fix GPU image to have new signing keys. [View in Git](https://github.com/mlrun/mlrun/pull/2030).
- Spark: Allow mounting v3io on driver but not executors. [View in Git](https://github.com/mlrun/mlrun/pull/2023).
- Tests: Send only string headers to align to new requests limitation. [View in Git](https://github.com/mlrun/mlrun/pull/2039).


### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.4)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.4)

## v1.0.3

### New and updated features
- Jupyter Image: Relax `artifact_path` settings and add README notebook. [View in Git](https://github.com/mlrun/mlrun/pull/2011).
- Images: Fix security vulnerabilities. [View in Git](https://github.com/mlrun/mlrun/pull/1997).

### Closed issues

- API: Fix projects leader to sync enrichment to followers. [View in Git](https://github.com/mlrun/mlrun/pull/2009).
- Projects: Fixes and usability improvements for working with archives. [View in Git](https://github.com/mlrun/mlrun/pull/2006).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.3)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.3)

## v1.0.2

### New and updated features

- Runtimes: Add java options to Spark job parameters. [View in Git](https://github.com/mlrun/mlrun/pull/1968).
- Spark: Allow setting executor and driver core parameter in Spark operator. [View in Git](https://github.com/mlrun/mlrun/pull/1973).
- API: Block unauthorized paths on files endpoints. [View in Git](https://github.com/mlrun/mlrun/pull/1967).
- Documentation: New quick start guide and updated docker install section. [View in Git](https://github.com/mlrun/mlrun/pull/1948).

### Closed issues
- Frameworks: Fix to logging the target columns in favor of model monitoring. [View in Git](https://github.com/mlrun/mlrun/pull/1929).
- Projects: Fix/support archives with project run/build/deploy methods. [View in Git](https://github.com/mlrun/mlrun/pull/1966).
- Runtimes: Fix jobs stuck in non-terminal state after node drain/preemption. [View in Git](https://github.com/mlrun/mlrun/pull/1964).
- Requirements: Fix ImportError on ingest to Azure. [View in Git](https://github.com/mlrun/mlrun/pull/1949).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.2)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.2)

## v1.0.0

### New and updated features

#### Feature store
- Supports snowflake as a datasource for the feature store.

#### Graph
- A new tab under **Projects | Models** named **Real-time pipelines** displays the real time pipeline graph, 
with a drill-down to view the steps and their details. [Tech Preview]

#### Projects
- Setting owner and members are in a dedicated **Project Settings** section.
- The **Project Monitoring** report has a new tile named **Consumer groups (v3io streams)** that shows the total number
   of consumer groups, with drill-down capabilities for more details.

#### Resource management
- Supports preemptible nodes.
- Supports configuring CPU, GPU, and memory default limits for user jobs.

#### UI
- Supports configuring pod priority.
- Enhanced masking of sensitive data.
- The dataset tab is now in the **Projects** main menu (was previously under the Feature store).

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.0)


## Open issues

| ID   | Description                                            | Workaround                                    | Opened |
| ---- | -------------------------------------------------------| --------------------------------------------- | ------ |
| 2223 | Cannot deploy a function when notebook names contain "." (ModuleNotFoundError) | Do not use "." in notebook name | 1.0.0  |
| 2199 | Spark operator job fails with default requests args.       | NA                                         | 1.0.0 |
| 1584 | Cannot run `code_to_function` when filename contains special characters | Do not use special characters in filenames | 1.0.0 |
| [2621](https://github.com/mlrun/mlrun/issues/2621) | Running a workflow whose project has `init_git=True`, results in Project error | Run `git config --global --add safe.directory '*'` (can substitute specific directory for *). | 1.1.0 |
| 2407 | Kafka ingestion service on empty feature set returns an error. | Ingest a sample of the data manually. This creates the schema for the feature set and then the ingestion service accepts new records. | 1.1.0 |

## Limitations


| ID   | Description                                                    | Workaround                           | Opened | 
| ---- | -------------------------------------------------------------- | ------------------------------------ | ----------|      
| 2014 | Model deployment returns ResourceNotFoundException (Nuclio error that Service <name> is invalid.) | Verify that all `metadata.labels` values are 63 characters or less. See the [Kubernetes limitation](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set). |  v1.0.0  |

 

## Deprecations

    
| In v.  | ID |Description                                                          |
|------ | ---- | --------------------------------------------------------------------|
| 1.0.0 |     | MLRun / Nuclio do not support python 3.6.                             |
| 1.3.0 |     | See [Deprecated APIs](#api-130).|
| |  | |