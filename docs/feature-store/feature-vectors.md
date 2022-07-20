(create-use-feature-vectors)=
# Creating and using feature vectors

You can define a group of features from different feature sets as a {py:class}`~mlrun.feature_store.FeatureVector`.  
Feature vectors are used as an input for models, allowing you to define the feature vector once, and in turn create and track the 
{ref}`datasets <retrieve-offline-data>` created from it or the online manifestation of the vector for real-time prediction needs.

The feature vector handles all the merging logic for you using an `asof merge` type merge that accounts for both the time and the entity.
It ensures that all the latest relevant data is fetched, without concerns about "seeing the future" or other types of common time related errors.

**In this section**

- [Creating a feature vector](#creating-a-feature-vector)
- [Using a feature vector](#using-a-feature-vector)
    
## Creating a feature vector

The feature vector object holds the following information:

- Name &mdash; the feature vector's name as will be later addressed in the store reference `store://feature_vectors/<project>/<feature-vector-name>` and the UI (after saving the vector).
- Description &mdash; a string description of the feature vector.
- Features &mdash; a list of features that comprise the feature vector.  
The feature list is defined by specifying the `<feature-set>.<feature-name>` for specific features or `<feature-set>.*` for all the feature set's features.
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
# so it will could be referenced by the `store://`
# and show in the UI
fv.save()
```

After saving the feature vector, it appears in the UI:

<img src="../_static/images/feature-store-vector-line.png" alt="feature-store-vector-line" width="800"/>

You can also view some metadata about the feature vector, including all the features, their types, a preview and statistics:

<img src="../_static/images/feature-store-vector-screen.png" alt="feature-store-vector-screen" width="800"/>

## Using a feature vector

After a feature vector is saved, it can be used to create both offline (static) datasets and online (real-time) instances to supply as input to a machine learning model.  

### Creating an offline feature vector

Use the feature store's {py:meth}`~mlrun.feature_store.get_offline_features` function to produce a `dataset` from the feature vector.
It creates the dataset (asynchronously if possible), saves it to the requested target, and returns a {py:class}`~mlrun.feature_store.OfflineVectorResponse`.  
Due to the async nature of this action, the response object contains an `fv_response.status` indicator that, once completed, could be directly turned into a `dataframe`, `parquet` or a `csv`.

`get_offline_features` expects to receive:

- **feature_vector** &mdash;  a feature vector store reference or object.
- **entity_rows** &mdash;  an optional dataframe that the features will be joined to.  
Defaults to the first feature set defined in the features vector's features list, and acts as the base for the vector's joins.
- **entity_timestamp_column** &mdash;  an optional specific timestamp column (from the defined features) to act as the base timestamp column.  
Defaults to the base feature set's timestamp entity.
- **target** &mdash;  a Feature Store target to write the results to.  
Defaults to return as a return value to the caller.
- **run_config** &mdash;  an optional function or a {py:class}`~mlrun.feature_store.RunConfig` to run the feature vector creation process in a remote function.
- **drop_columns** &mdash;  a list of columns to drop from the resulting feature vector.
Optional.
- **start_time** &mdash;  datetime, low limit of time needed to be filtered. Optional.
- **end_time** &mdash;  datetime, high limit of time needed to be filtered. Optional.

Here's an example of a new dataset from a parquet target:

```python
# Import the Parquet Target, so you can build your dataset from a parquet file
from mlrun.datastore.targets import ParquetTarget

# Get offline feature vector based on vector and parquet target
offline_fv = fstore.get_offline_features(feature_vector_name, target=ParquetTarget())

# Return dataset
dataset = offline_fv.to_dataframe()
```

Once an offline feature vector is created with a static target (such as {py:class}`~mlrun.datastore.targets.ParquetTarget()`) the reference to this dataset is saved as part of the feature vector's metadata and can now be referenced directly through the store as a function input using `store://feature-vectors/{project}/{feature_vector_name}`.

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

You can see a full example of using the offline feature vector to create an ML model in [part 2 of the end-to-end demo](./end-to-end-demo/02-create-training-model.html).

### Creating an online feature vector

The online feature vector provides real-time feature vectors to the model using the latest data available.

First create an `Online Feature Service` using {py:meth}`~mlrun.feature_store.get_online_feature_service`. Then feed the `Entity` of the feature vector to the service and receive the latest feature vector.

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