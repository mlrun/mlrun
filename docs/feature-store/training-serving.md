(training)=
# Training and serving using the feature store

When working on a new model we usually care about the experiment's reproducibility, results and how easy it is to re-create the proposed features and model environment for the serving task.  The feature store enables us to do all that in an easy and automated fashion.

After defining our [feature sets](transformations.md) and proposed a [feature vector](feature-vectors.md) for the experiment, the feature store will enable us to automatically extract a versioned **offline** static dataset based on the parquet target defined in the feature sets for training.

For serving, once we validated this is indeed the feature vector we want to use, we will use the **online** feature service, based on the nosql target defined in the feature set for real-time serving.

Using this feature store centric process, using one computation graph definition for a feature set, we receive an automatic online and offline implementation for our feature vectors, with data versioning both in terms of the actual graph that was used to calculate each data point, and the offline datasets that were created to train each model.

## How the solution should look like

<br><img src="../_static/images/feature-store-training-v2.png" alt="feature-store-training-graph" width="800"/><br>

