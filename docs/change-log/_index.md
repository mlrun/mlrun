(change-log)=
# Change log
- [v1.2.1](#v1-2-1)
- [v1.2.0](#v1-2-0)
- [v1.0.6](#v1-0-6)
- [v1.0.5](#v1-0-5)
- [v1.0.4](#v1-0-4)
- [v1.0.3](#v1-0-3)
- [v1.0.2](#v1-0-2)
- [v1.0.0](#v1-0-0)
- [Open issues](#open-issues)
- [Limitations](#limitations)
- [Deprecations](#deprecations)

## v1.2.1

### Closed issues
- 




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
- Support Redis as an online feature set for storey engine only. (See [Redis target store](../data-prep/ingest-data-fs.html#redis-target-store).)
- Support GCP objects as a data source for the feature store.
- Fully support ingesting with pandas engine, now equivalent to ingestion with `storey` engine (TechPreview):
   - Support DataFrame with multi-index.
   - Support mlrun steps when using pandas engine: `OneHotEncoder` , `DateExtractor`, `MapValue`, `Imputer` and `FeatureValidation`.
- Add new step: `DropFeature` for pandas and storey engines. (TechPreview)
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

- Feature store: Ingestion using pandas now takes the dataframe and creates indices out of the entity column (and removes it as 
   a column in this df). This could cause breakage for existing custom steps when using a pandas engine.

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



## v1.1.3

See [Closed issues](#closed-issues).

### See more
- [MLRun change log in GitHub]()
- [UI change log in GitHub]()

## v1.1.2

### New and updated features

**V3IO**
- v3io-py bumped to 0.5.19
- v3io-fs bumped to 0.1.15

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.2)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.2-rc3)

## v1.1.1

### New and updated features

#### API
- Supports workflow scheduling 

#### UI
- Projects: Supports editing model labels

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.1)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.1)


## v1.1.0

### New and updated features

#### API
-  MLRun scalability: Workers are used to handle the connection to the MLRun database and can be increased to improve handling of high workloads against the MLRun DB. You can configure the number of workers for an MLRun service, which is applied to the service's user-created pods. The default is 2. 
   - v1.1.0 cannot run on top of 3.0.x.
   - For Iguazio v <3.5.0 number of workers set to 1 by default. To change this number, contact support (helm-chart change required).
   - Multi-instance is not supported for MLrun running on SQLite.
-  Supports pipeline scheduling  
   
   
   
#### Feature store
- Supports S3, Azure, GCS targets when using Spark as an engine for the feature store
- Snowflake as datasource has a connector ID: `iguazio_platform`
- You can add a time-based filter condition when running `get_offline_feature` with a given vector. 


#### UI
Projects
- The Projects home page now has three tiles, Data, Jobs and Workflows, Deployment, that guide you through key capabilities of Iguazio, and provide quick access to common tasks.
- The Projects | Jobs | Monitor Jobs tab now displays the Spark UI URL.
- The information of the Drift Analysis tab is now displayed in the Model Overview.
- If there is an error, the error messages are now displayed in the Projects | Jobs | Monitor jobs tab.
Workflows
- The steps in Workflows are color-coded to identify their status: blue=running; green=completed; red=error.

#### Storey
- MLRun can write to parquet with flexible schema per batch for ParquetTarget: useful for inconsistent or unknown schema.

#### Documentation
- Added Azure and S3 examples to {ref}`Ingesting features with Spark <ingest-features-with-spark>`

### Closed issues



### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.0)

## v1.0.6

### Closed issues
- Import from mlrun fails with "ImportError: cannot import name dataclass_transform" (ML-2552)
   Workaround for previous releases:
   Install `pip install pydantic==1.9.2` after `align_mlrun.sh`. 
   [View in Git](https://github.com/mlrun/mlrun/pull/).
- MLRun FeatureSet was not not enriching with security context when running from the UI. [View in Git](https://github.com/mlrun/mlrun/pull/
- MlRun Accesskey presents as cleartext in the mlrun yaml, when the mlrun function is created by feature set 
   request from the UI. [View in Git](https://github.com/mlrun/mlrun/pull/).
   
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
- Images: Fix GPU image to have new signing keys. [View in Git](https://github.com/mlrun/mlrun/pull/2030).
- Spark: Allow mounting v3io on driver but not executors. <Backport 1.0.x> [View in Git](https://github.com/mlrun/mlrun/pull/2023).
- Tests: Send only string headers to align to new requests limitation  [View in Git](https://github.com/mlrun/mlrun/pull/2039).


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

- Runtimes: Add java options spark job parameters. [View in Git](https://github.com/mlrun/mlrun/pull/1968).
- Spark: Allow setting executor and driver core parameter in spark operator. [View in Git](https://github.com/mlrun/mlrun/pull/1973).
- API: Block unauthorized paths on files endpoints. [View in Git](https://github.com/mlrun/mlrun/pull/1967).
- Documentation: New quick start guide and update docker install section. [View in Git](https://github.com/mlrun/mlrun/pull/1948).

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

#### UI
- Supports configuring pod priority.
- Enhanced masking of sensitive data.
- The dataset tab is now in the Projects main menu (was previously under the Feature store).

#### Projects
- Setting owner and members are in a dedicated Project Settings section.
- The **Project Monitoring** report has a new tile named **Consumer groups (v3io streams)** that shows the total number
   of consumer groups, with drill-down capabilities for more details.


#### Resource management
- Supports preemptible nodes.
- Supports configuring CPU, GPU, and memory default limits for user jobs.

#### Graph
- A new tab under **Projects | Models** named **Real-time pipelines **that displays the real time pipeline graph, 
   with a drill-down to view the steps and their details. [Tech Preview]

### Closed issues

- 




### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.0)


## Open issues

| ID   | Description                                            | Workaround                                    | Opened |
| ---- | -------------------------------------------------------| --------------------------------------------- | ------ |
| 2489 | Cannot pickle a class inside an mlrun function         | Use cloudpickle instead of pickle             | 1.2.0  |
| 2223 | Cannot deploy a function when notebook names contain "." (ModuleNotFoundError) | Do not use "." in notebook name | 1.0.0  |
| 2199 | Spark operator job fails with default requests args       | NA                                         | 1.0.0 |
| 1584 | Cannot run `code_to_function` when filename contains special characters | Do not use special characters in filenames | 1.0.0 |
| 2637 | Running a workflow whose project has `init_git=True`, results in Project error | Run `git config --global --add safe.directory '*'` (can substitute specific directory for *). | 1.1.0 |
| 2407 | Kafka ingestion service on empty feature set returns an error. | Ingest a sample of the data manually, which creates the schema for the feature set and allow the ingestion service to accept new reocrds. | 1.1.0 |

## Limitations


| ID   | Description                                                    | Workaround                           | Opened | 
| ---- | -------------------------------------------------------------- | ------------------------------------ | ----------|      
| 2014 | Model deployment returns ResourceNotFoundException (Nuclio error that Service <name> is invalid.) | Verify that all `metadata.labels` values are 63 characters or less (Kubernetes limitation). |  v1.0.0  |
 

## Deprecations

    
| In v.  | ID |Description                                                          |
|------ | ---- | --------------------------------------------------------------------|
| 1.0.0 |     | MLRun / Nuclio do not support python 3.6                             |