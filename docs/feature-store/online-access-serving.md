
# Online access and serving


## Get online features

The online features are created ad-hoc using MLRun's feature store online feature service and are served from the **nosql** target for real-time performance needs.

To use it, first create an online feature service with the feature vector.

```python
import mlrun.feature_store as fstore

svc = fstore.get_online_feature_service(<feature vector name>)
```

After creating the service you can use the feature vector's Entity to get the latest feature vector for it.
Pass a list of `{<key name>: <key value>}` pairs to receive a batch of feature vectors.

```python
fv = svc.get([{<key name>: <key value>}])
```

## Incorporating to the serving model

MLRun enables you to easily serve your models using the [model server](../serving/serving-graph.md) ([example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb)).
It enables you to define a serving model class and the computational graph required to run your entire prediction pipeline and deploy it as serverless functions using [nuclio](https://github.com/nuclio/nuclio).

To embed the online feature service in your model server, all you need to do is create the feature vector service once when the model initializes and then use it to retrieve the feature vectors of incoming keys.

You can import ready-made classes and functions from our [function marketplace](https://github.com/mlrun/functions) or write your own.
As example of a scikit-learn based model server (taken from the MLRun feature store demo):

```python
from cloudpickle import load
import numpy as np
import mlrun
import os

class ClassifierModel(mlrun.serving.V2ModelServer):
    
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))
        
        # Setup FS Online service
        self.feature_service = mlrun.feature_store.get_online_feature_service('patient-deterioration')
        
        # Get feature vector statistics for imputing
        self.feature_stats = self.feature_service.vector.get_stats_table()
        
    def preprocess(self, body: dict, op) -> list:
        # Get patient feature vector 
        # from the patient_id given in the request
        vectors = self.feature_service.get([{'patient_id': patient_id} for patient_id in body['inputs']])
        
        # Impute inf's in the data to the feature's mean value
        # using the collected statistics from the Feature store
        feature_vectors = []
        for fv in vectors:
            new_vec = []
            for f, v in fv.items():
                if np.isinf(v):
                    new_vec.append(self.feature_stats.loc[f, 'mean'])
                else:
                    new_vec.append(v)
            feature_vectors.append(new_vec)
            
        # Set the final feature vector as our inputs
        # to pass to the predict function
        body['inputs'] = feature_vectors
        return body

    def predict(self, body: dict) -> list:
        """Generate model predictions from sample"""
        feats = np.asarray(body['inputs'])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
```

Which we can deploy with:

```python
# Create the serving function from our code above
fn = mlrun.code_to_function(<function_name>, 
                            kind='serving')

# Add a specific model to the serving function
fn.add_model(<model_name>, 
             class_name='ClassifierModel',
             model_path=<store_model_file_reference>)

# Enable MLRun's model monitoring
fn.set_tracking()

# Add the system mount to the function so
# it will have access to our model files
fn.apply(mlrun.mount_v3io())

# Deploy the function to the cluster
fn.deploy()
```

And test using:

```python
fn.invoke('/v2/models/infer', body={<key name>: <key value>})
```
