# Glossary

## MLRun Terms

| MLRun Terms              | Description                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FeatureSet               | A group of features that are ingested together and stored in logical group                                                                                                                      |
| FeatureVector            | A combination of multiple Features originating from different FeatureSets                                                                                                                       |
| HTTPRunDB                | API for wrapper to the internal DB in MLRun                                                                                                                                                     |
|hub                       | Used in code to reference the [MLRun Marketplace](../runtimes/load-from-marketplace)                                                                                                                                                     |
| MLRun Function           | An abstraction over the code, extra packages, runtime configuration and desired resources which allow execution in a local environment and on various serverless engines on top of K8s                                                                                                 |
| MLRun Marketplace        | A collection of pre-built MLRun functions avilable for usage                                                                                                                                    |
| MLRun project            | A logical container for all the work on a particular activity/application that include functions, workflow, artifacts, secrets, and more, and can be assigned to a specific group of users.                                                                    |
| Nuclio function          | Subtype of MLRun function that uses the Nuclio runtime for any generic real-time function                                                                                                            |
| Serving function         | Subtype of MLRun function that uses the Nuclio runtime specifically for serving ML models or real-time pipelines                                                                                     |
| storey                   | Asynchronous streaming library for real time event processing and feature extraction. Used in Iguazio's feature store and real-time pipelines                                                |
|                          |                                                                        

## Iguazio (V3IO) Terms
| Name                                       | Description          |   
|--------------------------------------------------|---------------------------------------------------------------------------| 
| Consumer group           | Set of consumers that cooperate to consume data from some topics                                                                                                                             |
| Key Value (KV) store     | Type of storage where data is stored by a specific key, allows for real-time lookups                                                                                                         |
| TSDB                     | Time series database: part of V3IO                                                                                                                                                           |
| V3IO                     | Iguazio real-time data layer, supports several formats including KV, Block, File, Streams, and more                                                                                                    |
| V3IO shard               | Uniquely identified data sets within a V3IO stream. Similar to a Kafka partition                                                                                                              |
| V3IO stream              | Streaming mechanism part of Iguazio's V3IO data layer. Similar to a Kafka stream                                                                                                              |

## Standard ML Terms
| Name                                       | Description          |   
|--------------------------------------------------|---------------------------------------------------------------------------| 
| Artifact                 | The input or output of experiments and model training runs that  can be used in many runs across the project.                                                                                  |
| DAG                      | Directed acyclic graph                                                                                                                                                                        |
| Feature engineering      | Apply domain knowledge and statistical techniques to raw data to extract more information out of data and improve performance of machine learning models                                      |
| EDA                      | Exploratory data analysis. Used by data scientists to understand dataset via cleaning, visualization, and statistical tests                                                                   |
| ML pipeline              | Pipeline of operations for machine learning. It can include loading data, feature engineering, feature selection, model training, hyperparameter tuning, model validation, and model deployment |
| Feature                  | Raw characteristic or numerical value of a dataset                                                                                                                                            |
| MLOps                    | Set of practices that reliably and efficiently deploys and maintains machine learning models in production. Combinination of Machine Learning and DevOps                                   |
| Dataframe                | Tabular representation of data, often using tools such as Pandas, Spark, or Dask                                                                                                              |

## ML Libraries / Tools


| Name              | Description                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dask                     | Flexible library for parallel computing in Python. Often used for data engieering, data science, and machine learning                                                                         |
| Keras                    | An open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.                                         |
| KubeFlow pipeline        | Platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers                                                                             |
| PyTorch                  | An open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing                                                 |
| Sklearn                  | Open source machine learning Python library. Used for modelling, pipelines, data transformations, feature engineering, and more.                                                              |
| Spark                    | Open source parallel processing framework for running large-scale data analytics applications across clustered computers. Often used for data engineering, data science, and machine learning |
| TensorFlow                    | An end-to-end open source platform for machine learning                                                                                                                                        |
| TensorBoard              |  TensorFlow’s visualization toolkit, used for tracking metrics like loss and accuracy, visualizing the model graph, viewign histograms of weights, biases, or other tensors as they change over time, etc. |
| XGBoost                  | Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Implements machine learning algorithms under the Gradient Boosting framework          |

[Back to top](#top)

<!--Add?
MPI - Message Passing Interface
-->

<!-- Really Specific - Maybe not for Glossary?	
ACCESS_KEY 	Some kind of authentication - no context for what this is
ctx 	Common abbreviation for context - should be evident from code
Event 	 part of streaming ... relation of event to row in FeatureSet - really specific?