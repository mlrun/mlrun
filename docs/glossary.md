(glossary)=
# Glossary

## MLRun terms

| MLRun terms        | Description                                                                                                                                                                                                                                                                                                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Feature set        | A group of features that are ingested together and stored in logical group. See {ref}`feature-sets`.                                                                                                           |
| Feature vector     | A combination of multiple features originating from different feature sets. See {ref}`create-use-feature-vectors`.                                                                                             |
| HTTPRunDB          | API for wrapper to the internal DB in MLRun. See {py:meth}`mlrun.db.httpdb.HTTPRunDB`.                                                                                                                         |
| hub                | Used in code to reference the {ref}`load-from-hub`.                                                                                                                                                            |
| MLRun function     | An abstraction over the code, extra packages, runtime configuration and desired resources which allow execution in a local environment and on various serverless engines on top of K8s. See {ref}`functions` . |
| MLRun Function hub | A collection of pre-built MLRun functions available for usage. See the {ref}`function hub documentation <load-from-hub>` and the [Function hub](https://www.mlrun.org/hub/).                                   |                                                 
| MLRun project      | A logical container for all the work on a particular activity/application that include functions, workflow, artifacts, secrets, and more, and can be assigned to a specific group of users. See {ref}`projects`.|
| mpijob             | One of the MLRun batch runtimes that runs distributed jobs and Horovod over the MPI job operator, used mainly for deep learning jobs. See {ref}`horovod`.                                                      |
| Nuclio function    | Subtype of MLRun function that uses the Nuclio runtime for any generic real-time function. See {ref}`nuclio-real-time-functions` and the [Nuclio documentation](https://docs.nuclio.io/en/stable/index.html).  |
| Serving function   | Subtype of MLRun function that uses the Nuclio runtime specifically for serving ML models or real-time pipelines. See {ref}`serving-graph`.                                                                    |
| storey             | Asynchronous streaming library for real time event processing and feature extraction. Used in Iguazio's feature store and real-time pipelines. See [storey.transformations - Graph transformations](./api/storey.transormations/storey.transformations.html).|  


## Iguazio (V3IO) terms

| Name                 | Description                                                                                         |   
|----------------------|-----------------------------------------------------------------------------------------------------| 
| Consumer group       | Set of consumers that cooperate to consume data from some topics.                                   |
| Key Value (KV) store | Type of storage where data is stored by a specific key, allows for real-time lookups.               |
| V3IO                 | Iguazio real-time data layer, supports several formats including KV, Block, File, Streams, and more. |
| V3IO shard           | Uniquely identified data sets within a V3IO stream. Similar to a Kafka partition.                   |
| V3IO stream          | Streaming mechanism part of Iguazio's V3IO data layer. Similar to a Kafka stream.                   |

## Standard ML terms
| Name                | Description                                                                                                                                                                                                                                                                          |   
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| Artifact            | A versioned output of a data processing or model training jobs, can be used as input for other jobs or pipelines in the project. There are various types of artifacts (file, model, dataset, chart, etc.) that incorporate useful metadata. See {ref}`artifacts`. |
| DAG                 | Directed acyclic graph, used to describe workflows/pipelines.                                                                                                                                                                                                                        |
| Feature engineering | Apply domain knowledge and statistical techniques to raw data to extract more information out of data and improve performance of machine. learning models                                                                                                                            |
| EDA                 | Exploratory data analysis. Used by data scientists to understand dataset via cleaning, visualization, and statistical tests.                                                                                                                                                         |
| ML pipeline         | Pipeline of operations for machine learning. It can include loading data, feature engineering, feature selection, model training, hyperparameter tuning, model validation, and model deployment.                                                                                     |
| Feature             | Data field/vector definition and metadata (name, type, stats, etc.). A dataset is a collection of features.                                                                                                                                                                          |
| MLOps               | Set of practices that reliably and efficiently deploys and maintains machine learning models in production. Combination of Machine Learning and DevOps.                                                                                                                              |
| Dataframe           | Tabular representation of data, often using tools such as Pandas, Spark, or Dask.                                                                                                                                                                                                    |

## ML libraries / tools


| Name              | Description                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dask                     | Flexible library for parallel computing in Python. Often used for data engineering, data science, and machine learning.                                                                         |
| Keras                    | An open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.                                         |
| KubeFlow pipeline        | Platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers. See [KubeFlow/KFP](https://github.com/kubeflow/pipelines).                   |
| PyTorch                  | An open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language. processing                                                 |
| SKLearn                  | Open source machine learning Python library. Used for modelling, pipelines, data transformations, feature engineering, and more.                                                              |
| Spark                    | Open source parallel processing framework for running large-scale data analytics applications across clustered computers. Often used for data engineering, data science, and machine learning. |
| TensorFlow               | A Google developed open-source software library for machine learning and deep learning.                                                                                                                            |
| TensorBoard              |  TensorFlow’s visualization toolkit, used for tracking metrics like loss and accuracy, visualizing the model graph, viewing histograms of weights, biases, or other tensors as they change over time, etc. |
| XGBoost                  | Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Implements machine learning algorithms under the Gradient Boosting framework.          |

[Back to top](#top)
