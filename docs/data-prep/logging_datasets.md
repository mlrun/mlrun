(logging_datasets)=
# Logging datasets 

Storing datasets is important in order to have a record of the data that was used to train 
models, as well as storing any processed data. MLRun comes with built-in support for the DataFrame format. MLRun not 
only stores the DataFrame, but it also provides information about the data, such as statistics.

The simplest way to store a dataset is with the following code:

``` python
context.log_dataset(key='my_data', df=df)
```

Where `key` is the name of the artifact and `df` is the DataFrame. By default, MLRun stores a short preview of 20 lines.
You can change the number of lines by changing the value of the `preview` parameter.

MLRun also calculates statistics on the DataFrame on all numeric fields. You can enable statistics regardless to the 
DataFrame size by setting the `stats` parameter to `True`.

## Logging a dataset from a job

The following example shows how to work with datasets from a job:

``` python
from os import path
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

# Ingest a data set into the platform
def get_data(context: MLClientCtx, source_url: DataItem, format: str = 'csv'):

    iris_dataset = source_url.as_df()

    target_path = path.join(context.artifact_path, 'data')
    # Optionally print data to your logger
    context.logger.info('Saving Iris data set to {} ...'.format(target_path))

    # Store the data set in your artifacts database
    context.log_dataset('iris_dataset', df=iris_dataset, format=format,
                        index=False, artifact_path=target_path)
```

You can run this function locally or as a job. For example, to run it locally:
``` python
from os import path
from mlrun import new_project, run_local, mlconf

project_name = 'my-project'
project_path = path.abspath('conf')
project = new_project(project_name, project_path, init_git=True)

# Target location for storing pipeline artifacts
artifact_path = path.abspath('jobs')
# MLRun DB path or API service URL
mlconf.dbpath = mlconf.dbpath or 'http://mlrun-api:8080'

source_url = 'https://s3.wasabisys.com/iguazio/data/iris/iris_dataset.csv'
# Run get-data function locally
get_data_run = run_local(name='get_data',
                         handler=get_data,
                         inputs={'source_url': source_url},
                         project=project_name,
                         artifact_path=artifact_path)
```

The dataset location is returned in the `outputs` field, therefore you can get the location by calling
`get_data_run.artifact('iris_dataset')` to get the dataset itself.


``` python
# Read your data set
get_data_run.artifact('iris_dataset').as_df()

# Visualize an artifact in Jupyter (image, html, df, ..)
get_data_run.artifact('confusion-matrix').show()
```

The dataset returned from the run result is of the `DataItem` type. It allows access to the data itself as a Pandas 
Dataframe by calling the `dataset.as_df()`. It also contains the metadata of the artifact, accessed by the using
`dataset.meta`. This artifact metadata object contains in it the statistics calculated, the schema of the dataset and 
other fields describing the dataset. For example, call `dataset.meta.stats` to obtain the data statistics. 
