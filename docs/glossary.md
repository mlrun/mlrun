# Glossary

## MLRun Terms

| MLRun Terms              | Definition                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FeatureSet               | A group of features that are ingested together and stored in logical group                                                                                                                      |
| FeatureVector            | A combination of multiple Features originating from different FeatureSets                                                                                                                       |
| MLRun Function           | An abstraction over the code, extra packages, runtime configuration and desired resources which allow execution in a local environment and on various serverless engines on top of K8s                                                                                                 |
| MLRun Marketplace        | A collection of pre-built MLRun functions avilable for usage                                                                                                                                    |
| MLRun Project            | A logical container for all the work on a particular activity/application that include functions, workflow, artifacts, secrets, and more, and can be assigned to a specific group of users.                                                                    |
| Nuclio Function          | Subtype of MLRun function that uses the Nuclio runtime for any generic real-time function                                                                                                            |
| Serving Function         | Subtype of MLRun function that uses the Nuclio runtime specifically for serving ML models or real-time pipelines                                                                                     |
| storey                   | Asynchronous streaming library for real time event processing and feature extraction. Used in Iguazio's feature store and real-time pipelines                                                |
|                          |                                                                        

## Iguazio (V3IO) Terms
| Name                                       | Description          |   
|--------------------------------------------------|---------------------------------------------------------------------------| 
| Consumer group           | Set of consumers that cooperate to consume data from some topics                                                                                                                             |
| Key Value (KV) store     | Type of storage where data is stored by a specific key, allows for real-time lookups                                                                                                         |
| TSDB                     | Time series database: part of V3IO                                                                                                                                                           |
| V3IO                     | Iguazio real-time data layer, supports several formats including KV, Block, File, Streams, and more                                                                                                    |
| V3IO Shard               | Uniquely identified data sets within a V3IO stream. Similar to a Kafka partition                                                                                                              |
| V3IO Stream              | Streaming mechanism part of Iguazio's V3IO data layer. Similar to a Kafka stream                                                                                                              |

## Standard ML Terms
| Name                                       | Description          |   
|--------------------------------------------------|---------------------------------------------------------------------------| 
| DAG                      | Directed acyclic graph                                                                                                                                                                        |
| Feature engineering      | Apply domain knowledge and statistical techniques to raw data to extract more information out of data and improve performance of machine learning models                                      |
| EDA                      | Exploratory data analysis. Used by data scientists to understand dataset via cleaning, visualization, and statistical tests                                                                   |
| ML Pipeline              | Pipeline of operations for machine learning. It can include loading data, feature engineering, feature selection, model training, hyperparameter tuning, model validation, and model deployment |
| Feature                  | Raw characteristic or numerical value of a dataset                                                                                                                                            |
| MLOps                    | Set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. Combinination of Machine Learning and DevOps                                |
| Dataframe                | Tabular representation of data, often using tools such as Pandas, Spark, or Dask                                                                                                              |

## ML Libraries / Tools


| MLRun Terms              | Definition                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dask                     | Flexible library for parallel computing in Python. Often used for data engieering, data science, and machine learning                                                                         |
| KubeFlow pipeline        | Platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers                                                                             |
| Sklearn                  | Open source machine learning Python library. Used for modelling, pipelines, data transformations, feature engineering, and more.                                                              |
| Spark                    | Open source parallel processing framework for running large-scale data analytics applications across clustered computers. Often used for data engineering, data science, and machine learning |
| XGBoost                  | Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Implements machine learning algorithms under the Gradient Boosting framework          |

[Back to top](#top)

<!--Add
TensorFlow, Keras, PyTorch, TensorBoard, HttpDB, Artifact, HUB, MPI - Message Passing Interface
-->

<!-- Really Specific - Maybe not for Glossary?	
ACCESS_KEY 	Some kind of authentication - no context for what this is
ctx 	Common abbreviation for context - should be evident from code
HTTPRunDB 	 API for wrapper to the internal DB in MLRun - really specific?
Event 	 part of streaming ... relation of event to row in FeatureSet - really specific?
hub 	Used in code to reference MLRun Marketplace -->