<a id="top"></a>

# Glossary of Terms


| MLRun Terms              | Definition                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FeatureSet               | Group of features that are ingested together and stored in logical group                                                                                                                      |
| FeatureVector            | Combination of multiple FeatureSets with ability to join, filter, etc.                                                                                                                        |
| MLRun Function           | Abstraction that allows for running Python code easily in local environment and on top of K8s                                                                                                 |
| MLRun Marketplace        | Collection of pre-built MLRun functions avilable for usage                                                                                                                                    |
| MLRun Project            | Logical separation of resources - includes model, jobs, real-time functions, datasets, feature sets, feature vectors, etc.                                                                    |
| Nuclio Function          | Subtype of MLRun function - uses Nuclio runtime for any generic real-time function                                                                                                            |
| Serving Function         | Subtype of MLRun function - uses Nuclio runtime specifically for serving ML models or real-time pipelines                                                                                     |
| storey                   | Asynchronous streaming library, for real time event processing and feature extraction. Used in Iguazio's feature store and real-time pipelines                                                |
|                          |                                                                                                                                                                                               |
| **Iguazio (V3IO) Terms** | **Definition**                                                                                                                                                                                |
| Consumer group           | Set of consumers which cooperate to consume data from some topics                                                                                                                             |
| Key Value (KV) store     | Type of storage where data is stored by a specific key - allows for real-time lookups                                                                                                         |
| TSDB                     | Time series database - part of V3IO                                                                                                                                                           |
| V3IO                     | Iguazio data layer - supports several formats including KV, Block, File, Streams, and more                                                                                                    |
| V3IO Shard               | Uniquely identified data sets within a V3IO stream. Similar to a Kafka partition                                                                                                              |
| V3IO Stream              | Streaming mechanism part of Iguazio's V3IO data layer. Similar to a Kafka stream                                                                                                              |
|                          |                                                                                                                                                                                               |
| **Standard ML Terms**    | **Definition**                                                                                                                                                                                |
| DAG                      | Directed acyclic graph                                                                                                                                                                        |
| Feature engineering      | Apply domain knowledge and statistical techniques to raw data to extract more information out of data and improve performance of machine learning models                                      |
| EDA                      | Exploratory data analysis. Allows data scientist to understand dataset via cleaning, visualization, and statistical tests                                                                     |
| ML Pipeline              | Pipeline of operations for machine learning - may include loading data, feature engineering, feature selection, model training, hyperparameter tuning, model validation, and model deployment |
| Feature                  | Raw characteristic or numerical value of a dataset                                                                                                                                            |
| MLOps                    | Set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. Combinination of Machine Learning and DevOps                                |
| Dataframe                | Tabular representation of data. Often using tools such as Pandas, Spark, or Dask                                                                                                              |
|                          |                                                                                                                                                                                               |
| **ML Libraries / Tools** | **Definition**                                                                                                                                                                                |
| Dask                     | Flexible library for parallel computing in Python. Often used for data engieering, data science, and machine learning                                                                         |
| KubeFlow pipeline        | Platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers                                                                             |
| Sklearn                  | Open source machine learning Python library. Used for modelling, pipelines, data transformations, feature engineering, and more.                                                              |
| Spark                    | Open source parallel processing framework for running large-scale data analytics applications across clustered computers. Often used for data engineering, data science, and machine learning |
| XGBoost                  | Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Implements machine learning algorithms under the Gradient Boosting framework          |

[Back to top](#top)
