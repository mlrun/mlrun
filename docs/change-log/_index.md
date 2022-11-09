(change-log)=
# Change log
- [v1.2.0](#v1-2-0)
- [v1.1.0](#v1-1-0)
- [Closed issues](#closed-issues)
- [Open issues](#open-issues)


## v1.2.0

**Release highlights**

- Artifact management improvements
- Support Redis as online feature set

**New features**

- V3IO
   - Supports upload of files larger that 5GB
- UI
   - The Labels in the Models > Overview tab can be edited 

**Documentation updates**
- The structure is updated to match the user workflow

**Significant bug fixes** 
(per component)

**Additional changes**
- Feature store: Ingestion using pandas now takes the dataframe and creates an index out of the entity column (and removes it from 
being a column in this df). This might cause breakage for existing custom steps when using a pandas engine.





[Full Change log](https://github.com/mlrun/mlrun/releases/tag/v1.1.1) and [UI change log](https://github.com/mlrun/ui/releases/tag/v1.1.1)



## v1.1.0

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

**Additional changes**

[Full Change log](https://github.com/mlrun/mlrun/releases/tag/v1.1.0) and [UI change log](https://github.com/mlrun/ui/releases/tag/v1.1.0)

## Open issues

`````{tab-set}
````{tab-item} v1.2
%https://jira.iguazeng.com/browse/ML-2516 
- Feature store is not backwards compatible

%https://jira.iguazeng.com/browse/ML-2396
**Cannot reproduce - should it be here??**
- When ingesting data with the feature store using the Spark engine without the default image, and adding an external requirements. Ingest fails.

%https://jira.iguazeng.com/browse/ML-2223
- Notebook name that contains a ".", the deploying function fails with a ModuleNotFoundError  

%https://jira.iguazeng.com/browse/ML-2199
- Spark operator job fails with default requests args

%https://jira.iguazeng.com/browse/ML-1584
- `code_to_function` fails when filename contains special characters
````

````{tab-item} v1.1

%https://jira.iguazeng.com/browse/ML-2669
- Flow fails when running a function after building the image.
   Workaround is to set the image parameter for the runtime, for example:</br>
   ```func = mlrun.code_to_function("func4", kind="job", handler="my_function", image="mlrun/mlrun", requirements=["pandas"])```
   This sets the created image in the image field (replaces the given value), and the run succeeds.

   Another option is to set the base image for the function:
   ```func.build_config(base_image="mlrun/mlrun")```

%https://jira.iguazeng.com/browse/ML-2664
- Cannot delete a model via tag

%https://jira.iguazeng.com/browse/ML-2396
**Cannot reproduce - should it be here??**
- When ingesting data with the feature store using the Spark engine without the default image, and adding an external requirements. Ingest fails.

%https://jira.iguazeng.com/browse/ML-2223
- Notebook name that contains a ".", the deploying function fails with a ModuleNotFoundError 

%https://jira.iguazeng.com/browse/ML-2199
- Spark operator job fails with default requests args

%https://jira.iguazeng.com/browse/ML-1584
- `code_to_function` fails when filename contains special characters

````

````{tab-item} v1.0


%https://jira.iguazeng.com/browse/ML-2664
- Cannot delete a model via tag

%https://jira.iguazeng.com/browse/ML-2396
**Cannot reproduce - should it be here??**
- When ingesting data with the feature store using the Spark engine without the default image, and adding an external requirements. Ingest fails.

%https://jira.iguazeng.com/browse/ML-2223
- Notebook name that contains a ".", the deploying function fails with a ModuleNotFoundError 

%https://jira.iguazeng.com/browse/ML-2199
- Spark operator job fails with default requests args

%https://jira.iguazeng.com/browse/ML-1584
- `code_to_function` fails when filename contains special characters

````
`````


## Closed issues


| Description                                                             | Closed in          |   
|-----------------------------------------------------------------------|--------------------|      
| Flow fails when running a function after building the image.          | v1.2               |
| Cannot delete a model via tag                                         | v1.2               |


