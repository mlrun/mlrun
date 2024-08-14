(model-monitoring-des)=
# Model monitoring description


## Architecture

<img src="../_static/images/model-monitoring.png" width="1100" >

</br>
</br>

When you call {py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`, you effectively deploy three components:
- application controller function: handles the monitoring processing and the triggers the apps that trigger the writer. The controller is a real-time Nuclio job whose frequency is determined by `base_period`. 
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. The monitored data is used to create real-time dashboards, detect drift, and analyze performance.
- writer function: writes the results and the metrics that output from the model monitoring applications to the databases, and outputs alerts according to the user configuration.

The model monitoring process flow starts with collecting operational data from a function in the model serving pod. The model 
monitoring stream pod forwards data to a Parquet database. MLRun supports integers, float, strings, images.
The controller periodically checks the Parquet DB for new data and forwards it to the relevant application. 
Each model monitoring application is a separate nuclio real-time function. Each one listens to a stream that is filled by 
the monitoring controller at each `base_period` interval.
The stream function examines the log entry, processes it into statistics which are then written to the statistics databases 
(parquet file, time series database and key value database). 
The monitoring stream function writes the Parquet files using a basic storey ParquetTarget. Additionally, there is a monitoring feature set that refers 
to the same target. You can use `get_offline_features` to read the data from that feature set. 

Read also how {ref}`model monitoring supports the gen AI model server<genai-mmonitor>`.

## Streaming platforms and credentials

Model monitoring supports open-source streaming platforms such as Kafka, TDEngine, MySQL (8.0 and higher), in addition to integration with the Iguazio AI platform V3IO data layer. 
Before you deploy the model monitoring or serving function, you need to {py:meth}`set the credentials <mlrun.projects.MlrunProject.set_model_monitoring_credentials>`. 

## Model monitoring applications

When you call `enable_model_monitoring` on a project, by default MLRun deploys te onitoring app, `HistogramDataDriftApplication`, which is 
tailored for classical ML models (not LLMs, gen AI, deep-learning models, etc.). It includes:
* Total Variation Distance (TVD) &mdash; The statistical difference between the actual predictions and the model's trained predictions.
* Hellinger Distance &mdash; A type of f-divergence that quantifies the similarity between the actual predictions, and the model's trained predictions.
* Kullback–Leibler Divergence (KLD) &mdash; The measure of how the probability distribution of actual predictions is different from the second model's trained reference probability distribution.

You can create your own model monitoring applications, for LLMs, gen AI, deep-learning models, etc., based on the class {py:meth}`mlrun.model_monitoring.applications.ModelMonitoringApplicationBaseV2`. 

Projects are used to group functions that use the same model monitoring application. You first need to create a project for aspecific application. 
Then you disable the default app, enable your customer app, and create and run the functions. 

See the:
- User flow example **in ??????**.
- Custom apps in **?????**.

The basic flow for classic ML and other models is the same, but the apps and the infer requests are different.




## Model and model monitoring endpoints 

Each unique combination of runtime function (a deployed Nuclio function) and model has an endpoint and a corresponding {py:meth}`model endpoint <mlrun.model_monitoring.api.get_or_create_model_endpoint>`. 
All model monitoring endpoints are presented in the UI with information about the actual inference, including data on the inputs, outputs, and results.
The Model Endpoints tab presents the overall metrics. From there you can select an endpoint and view the Overview, Features Analysis, and the Metrics tabs. 
Metrics are grouped under their applications. After you select the metrics and the timeframe, you get a histogram showing the number of occurrences/values range, and a timeline 
graph of the metric and the threshold. Any alerts are shown in the upper-right corner of the metrics box. For example:

<img src="../_static/images/mm_metrics.png" width="700" >


## Alerts and notifications

You can set up {ref}`alerts` to inform you about suspected and detected issues in the model monitoring functions. 
And you can use {ref}`notifications` to notify about alerts. 