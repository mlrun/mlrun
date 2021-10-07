# Feature-Store End-to-End Demo

This demo shows the usage of MLRun and the feature store. The demo will showcase:

- [**Data ingestion & preparation**](./01-ingest-datasources.ipynb)
- [**Model training & testing**](./02-create-training-model.ipynb)
- [**Model serving**](./03-deploy-serving-model.ipynb)

Fraud prevention specifically is a challenge as it requires processing raw transaction and events in real-time and being able to
quickly respond and block transactions before they occur. Consider, for example, a case where you would like to evaluate the
average transaction amount. When training the model, it is common to take a DataFrame and just calculate the average. However,
when dealing with real-time/online scenarios, this average has to be calculated incrementally.

In this demo we will learn how to **Ingest** different data sources to our **Feature Store**. Specifically, we will consider 2 types of data:  

- **Transactions**: Monetary activity between 2 parties to transfer funds.
- **Events**: Activity that done by the party, such as login or password change.

<img src="../../_static/images/feature_store_demo_diagram.png" width="600px" />

We will walk through creation of ingestion pipeline for each data source with all the needed preprocessing and validation. We will run the pipeline locally within the notebook and then launch a real-time function to **ingest live data** or schedule a cron to run the task when needed.

Following the ingestion, we will create a feature vector, select the most relevant features and create a final model. We will then deploy the model and showcase the feature vector and model serving.

```{toctree}
:maxdepth: 1

01-ingest-datasources
02-create-training-model
03-deploy-serving-model

```
