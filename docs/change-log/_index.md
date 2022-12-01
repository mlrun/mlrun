(change-log)=
# Change log
- [v1.2.0](#v1-2-0)
- [v1.1.3](#v1-1-3)
- [v1.1.2](#v1-1-2)
- [v1.1.1](#v1-1-1)
- [v1.1.0](#v1-1-0)
- [v1.0.0](#v1-0-0)
- [Closed issues](#closed-issues)
- [Open issues](#open-issues)
- [Limitations](#limitations)
- [Deprecations](#deprecations)


## v1.2.0

### New and updated features

**API**
- `deploy_function` has a new option of mock deployment that allows running the function locally
- Artifact management improvements: You can add and edit artifact tags
- Supports overwriting commands when using `project.build_function`

**Feature store**
- Supports Redis as an online feature set 
<br>See [Redis target store](../data-prep/ingest-data-fs.html#redis-target-store-tech-preview)
- The out-of-the-box feature store steps are available with the pandas engine
- Supports GCP objects as data source for the feature store

**UI**
- The Labels in the **Models > Overview** tab can be edited 
- Artifact management improvements: You can add and edit artifact tags in the UI

**Documentation**
- Restructure and new content

**Third party integrations**
- Supports Confluent Kafka (Tech Preview)

**Models**
- A new model server for HuggingFace models is available

**V3IO**
- Supports upload of files larger that 5GB

### Infrastructure improvements
- Refactor of artifacts' API endpoints. (The previous API will be deprecated in a future release.)

### Breaking changes

- Feature store: Ingestion using pandas now takes the dataframe and creates an index out of the entity column (and removes it from 
being a column in this df). This could cause breakage for existing custom steps when using a pandas engine.

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.2)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.2)



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

**API**
- Supports workflow scheduling 

**UI**
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

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.1.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.1.0)

## v1.0.0

### New and updated features

#### Feature store
- Supports snowflake as a datasource for the feature store

#### UI

- Supports configuring CPU, GPU, and memory default limits for user jobs
- Supports configuring pods priority and spot instances
- Enhanced masking of sensitive data
**Projects**
- Setting owner and members are in a dedicated Project Settings section
- Project Monitoring has a new Consumer Groups report


#### Resource management
- Supports preemptible nodes
- Add limits to all mlrun function, by default

#### Graph
- New screen shows real-time pipeline

### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.0.0)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.0.0)

## Closed issues

| ID   | Description                                                    | Workaround                  | Opened | Closed |
|----- | ---------------------------------------------------------------|----------------------------------|-----------|-----------|      
| 2778 | Getting Started demo: Artifacts dashboard does not display     | NA                               | 1.0.4  | 1.2.0  |
| 2669 | When running a function after building the image, the flow fails. | Set the image parameter for the runtime, for example: ```func = mlrun.code_to_function("func4", kind="job", handler="my_function", image="mlrun/mlrun", requirements=["pandas"])```. This sets the created image in the image field (replaces the given value)                          | 1.0.0     | 1.2.0    |
|      |                                                        | Another option is to set the base image for the function: ```func.build_config(base_image="mlrun/mlrun")```  |       |      | 
| 2055 | Can now delete a model via tag                                  |                                       | 1.0.0     |  1.2.0    |
| 2657 | Scheduled jobs on GKE fail with resource quota error           |                                       | 1.0.0     | 1.2.0 |
| 2421 | Demo: Clicking on artifact logged with ".html" artifact's dashboard does not give artifact details |          | 1.2.0     | 1.1.0 |
| 2104 | Deploying monitoring stream application results in `ResourceNotFoundException` |                             | 1.0.0     | 1.2.0  |
| 2330 | Deploy new function (job) should add source files to the new image |                                       | 1.0.0     | 1.2.0 |
| 2684 | Project's artifact-path should pass to remote runs   |                                       | 0.10.0     | 1.2.0 |
| 2679 | Kubeflow pipeline ParallelFor does not work on kfp 1.8.14      |  Use KFP < 1.8.14               | 1.1.0     | 1.2.0 |
| ???? | Fix legacy artifacts load when loading project                 |                                 | ????     | 1.1.3|
| 2865 | CLI: Timeout is applied when running pipeline                  | NA                              | 1.1.2 | 1.1.3|  
| 2873 | CLI: Supports overwriting schedule when creating scheduling workflow | NA                        | 1.1.2 | 1.1.3|  
| 2655 | Cluster went into a degraded mode, and projects not seen       | Create new project, old projects are returned. | 1.1.0 | 1.1.1 |
| 2518 | Mismatch between Nuclio and MLRun status screens               | NA                              | 1.1.0 | 1.1.1 |
| 2653 | **code_to_function fails with "BuildError: cannot convert notebook" on 3.2.3  **                 |       | 1.1.1 |                      
| 2062 | Job name needs to be validated to avoid DNS-1035 non-compliance | NA                             | 1.0.0 | 1.1.0 |
          


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
| ----- | -------------------------------------------------------------- | ------------------------------------ | ----------|      
| 2014 | Model deployment returns ResourceNotFoundException (Nuclio error that Service <name> is invalid.) | Verify that all `metadata.labels` values are 63 characters or less (Kubernetes limitation). |  v1.0.0  |
 

## Deprecations

| In v.  | ID |Description                                                          |
|------ | ---- | --------------------------------------------------------------------|
| 1.0.0 |     | MLRun / Nuclio do not support python 3.6                             |


%https://jira.iguazeng.com/browse/ML-2396
**Cannot reproduce - should it be here??**
- When ingesting data with the feature store using the Spark engine without the default image, and adding an external requirements. Ingest fails.
    