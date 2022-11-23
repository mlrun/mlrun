(change-log)=
# Change log
- [v1.2.0](#v1-2-0)
- [v1.1.0](#v1-1-0)
- [Closed issues](#closed-issues)
- [Open issues](#open-issues)


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

**UI**
- The Labels in the **Models > Overview** tab can be edited 
- Artifact management improvements: You can add and edit artifact tags in the UI

**Documentation**
- Documentation restructure and new content

**Third party integrations**
- Supports Confluent Kafka (Tech Preview)

**Models**
- A new model server for HuggingFace models is available

**V3IO**
- Supports upload of files larger that 5GB


### Infrastructure improvements
- Refactor of artifacts' API endpoints. (Deprecation of old endpoints planned in v1.4.0.)


### Breaking changes

- Feature store: Ingestion using pandas now takes the dataframe and creates an index out of the entity column (and removes it from 
being a column in this df). This could cause breakage for existing custom steps when using a pandas engine.





### See more
- [MLRun change log in GitHub](https://github.com/mlrun/mlrun/releases/tag/v1.2)
- [UI change log in GitHub](https://github.com/mlrun/ui/releases/tag/v1.2)

## Closed issues

| ID   | Description                                                    | Workaround                  | Opened | Closed |
|----- | ---------------------------------------------------------------|----------------------------------|-----------|-----------|      
| 2778 | Error in Getting Started demo: Artifacts dashboard does not display | NA                               | 1.0.4  | 1.2.0  |
| 2669 | Flow fails when running a function after building the image.          | Set the image parameter for the runtime, for example: ```func = mlrun.code_to_function("func4", kind="job", handler="my_function", image="mlrun/mlrun", requirements=["pandas"])```. This sets the created image in the image field (replaces the given value)                          | 1.0.0     | 1.2.0    |
|      |                                                        | Another option is to set the base image for the function: ```func.build_config(base_image="mlrun/mlrun")```  |       |      | 
| 2055 | Cannot delete a model via tag                                  |                                       | 1.0.0     |  1.2.0    |
| 2657 | Scheduled jobs on GKE fail with resource quota error           |                                       | 1.0.0     | 1.2.0 |
| 2421 | Demo: Clicking on artifact logged with ".html" artifact's dashboard gives error instead of artifact details |          | 1.2.0     | 1.1.0 |
| 2104 | `ResourceNotFoundException` when deploying monitoring stream application |                             | 1.0.0     | 1.2.0  |
| 2330 | Deploy new function (job) does not add source files to the new image |                                       | 1.0.0     | 1.2.0 |
| 2684 |  Project's artifact-path is not passed to remote runs   |                                       | 0.10.0     | 1.2.0 |
| 2679 |  Kubeflow pipeline ParallelFor does not work on kfp 1.8.14      |  Use KFP < 1.8.14               | 1.1.0     | 1.2.0 |
| 2669 | Run fails after deploying function without a defined image           |                                       | 1.0.4     | 1.2.0 |


## Open issues

| ID   | Description                                            | Workaround                                    | Opened  |
| ---- | -------------------------------------------------------| ------------------------------------ | ----------- |
| 2516 | Feature store is not backwards compatible              |  NA                                   | 1.2.0      |
| 2223 | Notebook names that contain "." cause the deploying function to fail with a ModuleNotFoundError | Do not use "." in notebook name | 1.0.0  |
| 2199 | Spark operator job fails with default requests args                                          | NA               | 1.0.0      |
| 1584 | `code_to_function` fails when filename contains special characters                           | Do not use special characters in filenames | 1.0.0      |
| 2637 | Project gets error when running a workflow whose project has `init_git=True`.                |NA                             | 1.1.0 |








## Limitations


| ID   | Description                                                    | Workaround                           | Opened | 
| ----- | -------------------------------------------------------------- | ------------------------------------ | ----------|      
| 2014 | Model deployment fails with ResourceNotFoundException (Nuclio error that Service <name> is invalid.) | Verify that all `metadata.labels` values are 63 characters or less (Kubernetes limitation). |  v1.0.0  |
 

## Deprecations

| In v.  | ID |Description                                                          |
|------ | ---- | --------------------------------------------------------------------|
| 1.2  | 1188 |API: `init_function` is deprecated. Use `fn.apply(mlrun.auto_mount)` in the pipeline itself |


%https://jira.iguazeng.com/browse/ML-2396
**Cannot reproduce - should it be here??**
- When ingesting data with the feature store using the Spark engine without the default image, and adding an external requirements. Ingest fails.
    
[## v1.1.0

**Release highlights**
scalable mlrun-api
Support various targets when using spark as an engine for the feature store
Add filter to the get_offline_features
Scheduling Pipelines
Security Bug Fixes : ML-2175, ML-2164


**New features**

- Feature store
   - Support fetching data from a feature vector using spark engine
   - You can add a time-based filter condition when running `get_offline_feature` with a given vector. You can also filter with the query argument on all the other features as relevant.
- MLRun 
   Workers are used to handle the connection to the MLRun database and can be increased to improve handling of high workloads 
   against the MLRun DB. You can configure the number of workers for an MLRun service (MLRun Service page > Common Parameters), 
   which is applied to the service's user-created pods. The default is 2. 

   ```{admonition} Warning
   Do not change the MLRun service parameters unless Iguazio support recommends that you change them.
   ```
- UI
   - The steps in projects workflows are color-coded to identify their status: blue=running; green=completed; red=error.

**Documentation updates**

**Significant bug fixes**

See [Issues closed in v1.2](#issues-closed-in-v1-2)

**Full Change log**

[Here](https://github.com/mlrun/mlrun/releases/tag/v1.1.0) and [UI change log](https://github.com/mlrun/ui/releases/tag/v1.1.0)]: #
   