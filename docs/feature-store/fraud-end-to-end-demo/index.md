# Feature-Store Transactions Fraud End-To-End Demo

This demo shows the usage of MLRun and the feature store. The demo will showcase:
- [**Data ingestion & preparation**](./1-data-ingestion.ipynb)
- [**Model training & testing**](./2-model-training.ipynb)
- [**Real-time data & model pipeline**](./3-model-serving.ipynb)

These steps are the first key steps in the MLRun architecture:

![MLRun Architecture](../../_static/images/mlrun-architecture.png)

Financial transactions today are being done online or via digital means which could be duplicated or stolen.
To identify such transactions, all the financial activities are being closely monitored by the banks.

There are many types of fraud that could be done, and such many different algorithms to solve the spot them.
In this use case we will use an ensemble of three different classical machine learning algorithms to spot fraudulent transactions in real time using MLRun's Feature Store.

We will use PaySim based dataset which contains different transaction types amongst different users and account events such as logins and password changes.

We will process the data as three different FeatureSets, each with a special preprocessing step.
We will create a Feature Vector combining all three, create a dataset and feed it to a machine learning training function.
Then we will deploy the models as an ensemble to our kubernetes cluster and test it.

```{toctree}
:maxdepth: 1

1-data-ingestion
2-model-training
3-model-serving
```