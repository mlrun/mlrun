(create-use-feature-vectors)=
# Creating and using feature vectors

You can define a group of features from different feature sets as a {py:class}`~mlrun.feature_store.FeatureVector`.  
Feature vectors are used as an input for models, allowing you to define the feature vector once, and in turn create and track the 
{ref}`datasets <retrieve-offline-data>` created from it or the online manifestation of the vector for real-time prediction needs.

The feature vector handles all the merging logic for you using an `asof merge` type merge that accounts for both the time and the entity.
It ensures that all the latest relevant data is fetched, without concerns about "seeing the future" or other types of common time-related errors.

**In this section**

- [Creating a feature vector](#creating-a-feature-vector)
- [Using a feature vector](#using-a-feature-vector)
    
## Creating a feature vector

The feature vector object holds the following information:

- Name &mdash; the feature vector's name as will be later addressed in the store reference `store://feature_vectors/<project>/<feature-vector-name>` and the UI (after saving the vector).
- Description &mdash; a string description of the feature vector.
- Features &mdash; a list of features that comprise the feature vector.  
The feature list is defined by specifying the `<feature-set>.<feature-name>` for specific features or `<feature-set>.*` for all of the feature set's features.
- Label feature &mdash; the feature that is the label for this specific feature vector, as a `<feature-set>.<feature-name>` string specification.

Example of creating a feature vector:

```python
import mlrun.feature_store as fstore

# Feature vector definitions
feature_vector_name = 'example-fv'
feature_vector_description = 'Example feature vector'
features = ['data_source_1.*', 
            'data_source_2.feature_1', 
            'data_source_2.feature_2',
            'data_source_3.*']
label_feature = 'label_source_1.label_feature'

# Feature vector creation
fv = fstore.FeatureVector(name=feature_vector_name,
                          features=features,
                          label_feature=label_feature,
                          description=feature_vector_description)

# Save the feature vector in the MLRun DB
# so it can be referenced by the `store://`
# and show in the UI
fv.save()
```

After saving the feature vector, it appears in the UI:

<img src="../_static/images/feature-store-vector-line.png" alt="feature-store-vector-line" width="800"/>

You can also view some metadata about the feature vector, including all the features, their types, a preview, and statistics:

<img src="../_static/images/feature-store-vector-screen.png" alt="feature-store-vector-screen" width="800"/>

## Using a feature vector

After a feature vector is saved, it can be used to create both offline (static) datasets and online (real-time) instances to supply as input to a machine learning model.  

### Using an offline feature vector

Use the feature store's {py:meth}`~mlrun.feature_store.get_offline_features` function to produce a `dataset` from the feature vector.
It creates the dataset (asynchronously if possible), saves it to the requested target, and returns an {py:class}`~mlrun.feature_store.OfflineVectorResponse`.  
Due to the async nature of this action, the response object contains an `fv_response.status` indicator that, once completed, could be directly turned into a `dataframe`, `parquet` or a `csv`.

`get_offline_features` supports Storey, Dask, Spark Operator, and Remote Spark.

`get_offline_features` expects to receive:

- **feature_vector** &mdash; A feature vector store reference or object.
- **entity_rows** &mdash; (optional) A dataframe that the features will be joined to. 
Defaults to the first feature set defined in the features vector's features list, and acts as the base for the vector's joins.
- **entity_timestamp_column** &mdash; (optional) A specific timestamp column (from the defined features) to act as the base timestamp column. 
Defaults to the base feature set's timestamp entity.
- **target** &mdash; A Feature Store target to write the results to.  
Defaults to return as a return value to the caller.
- **run_config** &mdash; (optional) A function or a {py:class}`~mlrun.feature_store.RunConfig` to run the feature vector creation process in a remote function.
- **drop_columns** &mdash; (optional) A list of columns to drop from the resulting feature vector.
- **start_time** &mdash; (optional) Datetime, low limit of time needed to be filtered. 
- **end_time** &mdash; (optional) Datetime, high limit of time needed to be filtered. 
- **with_indexes**    return vector with index columns and timestamp_key from the feature sets. Default is False.
- **update_stats** &mdash; update features statistics from the requested feature sets on the vector. Default is False.
- **engine** &mdash; processing engine kind ("local", "dask", or "spark")
- **engine_args** &mdash; kwargs for the processing engine.
- **query** &mdash; The query string used to filter rows on the output.
- **spark_service** &mdash; Name of the spark service to be used (when using a remote-spark runtime)
- **order_by** &mdash; Name or list of names to order by. The name or the names in the list can be the feature name or the alias of the 
feature you pass in the feature list.
- **timestamp_for_filtering** &mdash; (optional) Used to configure the columns that a time-based filter filters by. By default, the time-based filter is executed using the timestamp_key of each feature set.
Specifying the `timestamp_for_filtering` param overwrites this default: if it's str it specifies the timestamp column to use in all the feature sets. If it's a dictionary ({<feature set name>: <timestamp column name>, …}) it indicates the timestamp column name 
for each feature set. The time filtering is performed on each feature set (using `start_time` and `end_time`) before the merge process.

You can create a feature vector that comprises different feature sets, while joining the data based on specific fields and not the entity. 
For example:
- Feature set A is a transaction feature set and one of the fields is email.
- Feature set B is feature set with the fields email and count distinct.
You can build a feature vector that comprises fields in feature set A and get the count distinct for the email from feature set B. 
The join in this case is based on the email column.

Here's an example of a new dataset from a Parquet target:

```python
# Import the Parquet Target, so you can build your dataset from a parquet file
from mlrun.datastore.targets import ParquetTarget

# Get offline feature vector based on vector and parquet target
offline_fv = fstore.get_offline_features(feature_vector_name, target=ParquetTarget())

# Return dataset
dataset = offline_fv.to_dataframe()
```

After you create an offline feature vector with a static target (such as {py:class}`~mlrun.datastore.targets.ParquetTarget()`) the 
reference to this dataset is saved as part of the feature vector's metadata and can be referenced directly through the store 
as a function input using `store://feature-vectors/{project}/{feature_vector_name}`.

For example:

```python
fn = mlrun.import_function('hub://sklearn-classifier').apply(auto_mount())

# Define the training task, including the feature vector and label
task = mlrun.new_task('training', 
                      inputs={'dataset': f'store://feature-vectors/{project}/{feature_vector_name}'},
                      params={'label_column': 'label'}
                     )

# Run the function
run = fn.run(task)
```

See a full example of using the offline feature vector to create an ML model in [part 2 of the end-to-end demo](./end-to-end-demo/02-create-training-model.html).

You can use `get_offline_features` for a feature vector whose data is not ingested. See 
[Create a feature set without ingesting its data](./feature-sets.html#create-a-feature-set-without-ingesting-its-data).

#### Using joins in an offline feature vector

You can create a join for:
- Feature sets that have a common entity
- Feature sets that do not have a common entity

**Feature sets that have a common entity**

In this case, the join is performed on the common entity.

```
employees_set_entity = fs.Entity("id")
employees_set = fs.FeatureSet(
    "employees",
    entities=[employees_set_entity],
)
employees_set.set_targets(targets=["parquet"], with_defaults=False)
fs.ingest(employees_set, employees)

mini_employees_set = fs.FeatureSet(
    "mini-employees",
    entities=[employees_set_entity],
    },
)
mini_employees_set.set_targets(targets=["parquet"], with_defaults=False)
fs.ingest(mini_employees_set, employees_mini)

features = ["employees.name as n", "mini-employees.name as mini_name"]

vector = fs.FeatureVector(
    "mini-emp-vec", features, description="Employees feature vector"
)
vector.save()

resp = fs.get_offline_features(
    vector,
    engine_args=engine_args,
    with_indexes=True,
)
```

**Feature sets that do not have a common entity**

In this case, you define the relations between the features set with the argument: ` relations={column_name(str): Entity}`</br>
and you include this dictionary when initializing the feature set. 

```
departments_set_entity = fs.Entity("d_id")
departments_set = fs.FeatureSet(
    "departments",
    entities=[departments_set_entity],
)

departments_set.set_targets(targets=["parquet"], with_defaults=False)
fs.ingest(departments_set, departments)

employees_set_entity = fs.Entity("id")
employees_set = fs.FeatureSet(
    "employees",
    entities=[employees_set_entity],
    relations={"department_id": departments_set_entity},  # dictionary where the key is str identifying a column/feature on this feature-set, and the dictionary value is an Entity object on another feature-set
)
employees_set.set_targets(targets=["parquet"], with_defaults=False)
fs.ingest(employees_set, employees)
features = ["employees.name as emp_name", "departments.name as dep_name"]

vector = fs.FeatureVector(
    "employees-vec", features, description="Employees feature vector"
)

resp = fs.get_offline_features(
    vector,
    engine_args=engine_args,
    with_indexes=False,
)
```

### Using an online feature vector

The online feature vector provides real-time feature vectors to the model using the latest data available.

First create an `Online Feature Service` using {py:meth}`~mlrun.feature_store.get_online_feature_service`. Then feed the `Entity` of the 
feature vector to the service and receive the latest feature vector.

To create the {py:class}`~mlrun.feature_store.OnlineVectorService` you only need to pass it the feature vector's store reference.

```python
import mlrun.feature_store as fstore

# Create the Feature Vector Online Service
feature_vector = 'store://feature-vectors/{project}/{feature_vector_name}'
svc = fstore.get_online_feature_service(feature_vector)
```

The online feature service supports value imputing (substitute NaN/Inf values with statistical or constant value). You 
can set the `impute_policy` parameter with the imputing policy, and specify which constant or statistical value will be used
instead of NaN/Inf value. This can be defined per column or for all the columns (`"*"`).
The replaced value can be a fixed number for constants or `$mean`, `$max`, `$min`, `$std`, `$count` for statistical values.
`"*"` is used to specify the default for all features, for example: 

    svc = fstore.get_online_feature_service(feature_vector, impute_policy={"*": "$mean", "age": 33})


To use the online feature service you need to supply a list of entities you want to get the feature vectors for.
The service returns the feature vectors as a dictionary of `{<feature-name>: <feature-value>}` or simply a list of values as numpy arrays.

For example:

```python
# Define the wanted entities
entities = [{<feature-vector-entity-column-name>: <entity>}]

# Get the feature vectors from the service
svc.get(entities)
```

The `entities` can be a list of dictionaries as shown in the example, or a list of lists where the values in the internal 
list correspond to the entity values (e.g. `entities = [["Joe"], ["Mike"]]`). The `.get()` method returns a dict by default. 
If you want to return an ordered list of values, set the `as_list` parameter to `True`. The list input is required by many ML 
frameworks and this eliminates additional glue logic.  

See a full example of using the online feature service inside a serving function in [part 3 of the end-to-end demo](./end-to-end-demo/03-deploy-serving-model.html).