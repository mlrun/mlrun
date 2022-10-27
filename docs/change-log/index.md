(change-log)=
# Change log

## v1.1.1

- V3IO: Supports upload of files larger that 5GB
- UI: The Labels in the Models > Overview tab can be edited 
- Documentation: The structure is updated to match the user workflow

[Full Change log](https://github.com/mlrun/mlrun/releases/tag/v1.1.1) and [UI change log](https://github.com/mlrun/ui/releases/tag/v1.1.1)



## v1.1.0

- Feature store: Support fetching data from a feature vector using spark engine
- Feature store: You can add a time-based filter condition when running `get_offline_feature` with a given vector. You can also filter with the query argument on all the other features as relevant.
- Workers are used to handle the connection to the MLRun database and can be increased to improve handling of high workloads 
   against the MLRun DB. You can configure the number of workers for an MLRun service (MLRun Service page > Common Parameters), 
   which is applied to the service's user-created pods. The default is 2. 

   ```{admonition} Warning
   Do not change the MLRun service parameters unless Iguazio support recommends that you change them.
   ```
- UI: The steps in projects workflows are color-coded to identify their status: blue=running; green=completed; red=error.


[Full Change log](https://github.com/mlrun/mlrun/releases/tag/v1.1.0) and [UI change log](https://github.com/mlrun/ui/releases/tag/v1.1.0)