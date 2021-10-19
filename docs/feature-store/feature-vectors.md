# Feature Vectors

In MLRun, you can define a group of features from different feature sets as a {py:class}`~mlrun.feature_store.FeatureVector`.  
Feature Vectors are used as an input for models, allowing you to define the feature vector once and in turn create and track the datasets created from it or the online manifestation of the vector for real-time prediction needs.

The feature vector will handle all the merging logic for you using an `asof merge` type merge which accounts for both the time and the entity.
Thus making sure that all the latest relevant data will be fetched and without worrying about "seeing the future" or other types of common time related errors.

## Creating a Feature Vector

The Feature Vector object holds the following information:

- **Name**: the feature vector's name as would be later addressed in the store reference `store://feature_vectors/<project>/<feature-vector-name>` and the UI.  (after saving the vector)
- **Description**: a string description of the feature vector.
- **Features**: a list of features that comprises the feature vector.  
The feature list is defined by specifying the `<feature-set>.<feature-name>` for specific features or `<feature-set>.*` for all the feature set's features.
- **Label feature**: the feature that is the label for this specific feature vector. as a `<feature-set>.<feature-name>` string specification.

a Feature Vector creation example:

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
# so it will could be referenced by the `store://`
# and show in the UI
fv.save()
```

Once saving the feature vector, it will appear in the UI:

<img src="../_static/images/feature-store-vector-line.png" alt="feature-store-vector-line" width="800"/>

And we can also view some metadata about the feature vector, including all the features, their types, a preview and statistics:

<img src="../_static/images/feature-store-vector-screen.png" alt="feature-store-vector-screen" width="800"/>

## Using the Feature Vector

After a feature vector has been defined, it can be used to create both "offline" (static) datasets and "Online" real-time instances to supply as input to a machine learning model.  

### Creating an Offline Feature Vector

To produce a `dataset` from the feature vector we will use the feature store's {py:meth}`~mlrun.feature_store.get_offline_features` function.
It will create the dataset for us (asynchronously if possible), save it to the requested target and return a {py:class}`~mlrun.feature_store.OfflineVectorResponse`.  
Due to the async nature of this action, the response object contains an `fv_response.status` indicator that once completed could be directly turned into a `dataframe`, `parquet` or a `csv`.


`get_offline_features` expects to receive:

- **feature_vector** - a feature vector store reference or object.
- **entity_rows** - an optional dataframe on which the features will be joined to.  
Defaults to the first feature set defined in the features vector's features list will act as the base for the vector's joins.
- **entity_timestamp_column** - an optional specific timestamp column (from the defined features) to act as the base timestamp column.  
Defaults to the base feature set's timestamp entity.
- **target** - a Feature Store target to write the results to.  
Defaults to return as a return value to the caller.
- **run_config** - a Function or a {py:class}`~mlrun.feature_store.RunConfig` to run the feature vector creation process in a remote function.
Optional.
- **drop_columns** - a list of columns to drop from the resulting feature vector.
Optional.
- **start_time** - datetime, low limit of time needed to be filtered. Optional.
- **end_time** - datetime, high limit of time needed to be filtered. Optional.

As an example, lets create a new dataset and save it as a parquet file:

```python
# Import the Parquet Target so we can directly save our dataset as a file
from mlrun.datastore.targets import ParquetTarget

# Get offline feature vector/
# will return a pandas dataframe and save the dataset to parquet
# so a training job could train on it
offline_fv = fstore.get_offline_features(feature_vector_name, target=ParquetTarget())

# View dataset
dataset = offline_fv.to_dataframe()
```

Once an offline feature vector was created with a static target (such as {py:class}`~mlrun.datastore.targets.ParquetTarget()`) the reference to this dataset will be saved as part of the feature vector's metadata and could now be referenced directly through the store as a function input using `store://feature-vectors/{project}/{feature_vector_name}`.

for example:

```python
fn = mlrun.import_function('hub://sklearn-classifier').apply(auto_mount())

# Define the training task, including our feature vector and label
task = mlrun.new_task('training', 
                      inputs={'dataset': f'store://feature-vectors/{project}/{feature_vector_name}'},
                      params={'label_column': 'label'}
                     )

# Run the function
run = fn.run(task)
```

You can see a full example of using the offline feature vector to create an ML model in [part 2 of our end-to-end demo](./end-to-end-demo/02-create-training-model.ipynb).

### Creating an Online Feature Vector

The online feature vector is intended to provide real-time feature vectors to our model using the latest data available.

To do this we need to first create an `Online Feature Service` using {py:meth}`~mlrun.feature_store.get_online_feature_service`. Then we can feed the `Entity` of our feature vector to the service and receive the latest feature vector.

To create the {py:class}`~mlrun.feature_store.OnlineVectorService` you only need to pass it the feature vector's store reference.

```python
import mlrun.feature_store as fstore

# Create the Feature Vector Online Service
feature_vector = 'store://feature-vectors/{project}/{feature_vector_name}'
svc = fstore.get_online_feature_service(feature_vector)
```

The online feature service support value imputing (substitute NaN/Inf values with statistical or constant value), you 
can set the `impute_policy` parameter with the imputing policy, and specify which constant or statistical value will be used
instead of NaN/Inf value, this can be defined per column or for all the columns (`"*"`).
the replaced value can be fixed number for constants or $mean, $max, $min, $std, $count for statistical values.
"*" is used to specify the default for all features, example: 

    svc = fstore.get_online_feature_service(feature_vector, impute_policy={"*": "$mean", "age": 33})


To use the online feature service we will need to supply him with a list of entities we would like to get the feature vectors for.
The service will return us the feature vectors as a dictionary of `{<feature-name>: <feature-value>}` or simply a list of values as numpy arrays.

for example:

```python
# Define the wanted entities
entities = [{<feature-vector-entity-column-name>: <entity>}]

# Get the feature vectors from the service
svc.get(entities)
```

The `entities` can be a list of dictionaries as shown in the example, or a list of list where the values in the internal 
list correspond to the entity values (e.g. `entities = [["Joe"], ["Mike"]]`). the `.get()` method returns a dict by default
, if we want to return an ordered list of values we set the `as_list` parameter to `True`, list input is required by many ML 
frameworks and this eliminates additional glue logic.  

You can see a full example of using the online feature service inside a serving function in [part 3 of our end-to-end demo](./end-to-end-demo/03-deploy-serving-model.ipynb).
