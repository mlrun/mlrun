(training-serving)=
# Serving with the feature store

**In this section**
- [Get online features](#get-online-features)
- [Incorporating to the serving model](#incorporating-to-the-serving-model)

## Get online features

The online features are created using MLRun's feature store online feature service and are served from the **NoSQL** target for real-time performance needs.

To use it, first create an online feature service with the feature vector.

```python
import mlrun.feature_store as fstore

# Create the Feature Vector Online Service
feature_vector = "store://feature-vectors/{project}/{feature_vector_name}"
fvec = fstore.get_feature_vector(feature_vector)
svc = fvec.get_online_feature_service()
```

After creating the service, you can use the feature vector's entity to get the latest feature vector for it.
Pass a list of `{"<key name>": "<key value>"}` pairs to receive a batch of feature vectors.

```python
fv = svc.get([{"<key name>": "<key value>"}])
```

## Incorporating to the serving model

You can serve your models using the {ref}`serving-graph`. (See a [V2 Model Server (SKLearn) example](https://github.com/mlrun/functions/blob/master/v2_model_server/v2_model_server.ipynb).)
You define a serving model class and the computational graph required to run your entire prediction pipeline, and deploy it as a serverless function using [Nuclio](https://github.com/nuclio/nuclio).

To embed the online feature service in your model server, just create the feature vector service once when the model initializes, and then use it to retrieve the feature vectors of incoming keys.

You can import ready-made classes and functions from the MLRun [Function Hub](https://www.mlrun.org/hub/) or write your own.
As example of a scikit-learn based model server:
<!--- (taken from the [feature store demo](./end-to-end-demo/03-deploy-serving-model.html#define-model-class)) --->

```python
from cloudpickle import load
import numpy as np
import mlrun
import mlrun.feature_store as fstore
import os


class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

        # Setup FS Online service
        self.feature_service = fstore.get_feature_vector(
            "store://patient-deterioration"
        ).get_online_feature_service()

        # Get feature vector statistics for imputing
        self.feature_stats = self.feature_service.vector.get_stats_table()

    def preprocess(self, body: dict, op) -> list:
        # Get patient feature vector
        # from the patient_id given in the request
        vectors = self.feature_service.get(
            [{"patient_id": patient_id} for patient_id in body["inputs"]]
        )

        # Impute inf's in the data to the feature's mean value
        # using the collected statistics from the Feature store
        feature_vectors = []
        for fv in vectors:
            new_vec = []
            for f, v in fv.items():
                if np.isinf(v):
                    new_vec.append(self.feature_stats.loc[f, "mean"])
                else:
                    new_vec.append(v)
            feature_vectors.append(new_vec)

        # Set the final feature vector as the inputs
        # to pass to the predict function
        body["inputs"] = feature_vectors
        return body

    def predict(self, body: dict) -> list:
        """Generate model predictions from sample"""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
```

Which you can deploy with:

```python
project = mlrun.get_or_create_project("prediction")
# Create the serving function from the code above
fn = project.set_function(name="<function_name>", kind="serving")

# Add a specific model to the serving function
fn.add_model(
    "<model_name>",
    class_name="ClassifierModel",
    model_path="<store_model_file_reference>",
)

# Add the system mount to the function so
# it will have access to the model files
fn.apply(mlrun.mount_v3io())

# Deploy the function to the cluster
fn.deploy()
```

And test using:

```python
fn.invoke("/v2/models/infer", body={"<key name>": "<key value>"})
```