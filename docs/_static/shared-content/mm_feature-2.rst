
The model monitoring process flow starts with collecting operational data from a function in the model serving pod. The model 
monitoring stream pod forwards data to a Parquet database. 
The controller periodically checks the Parquet DB for new data and forwards it to the relevant application. 
Each monitoring application is a separate nuclio real-time job. Each one listens to a stream that is filled by 
the monitoring controller at each `base_period` interval.
The stream function examines 
the log entry, processes it into statistics which are then written to the statistics databases (parquet file, time series database and key value database). 
The monitoring stream function writes the Parquet files using a basic storey ParquetTarget. Additionally, there is a monitoring feature set that refers 
to the same target. You can use `get_offline_features` to read the data from that feature set. 

When you enable model monitoring ({py:meth}`~mlrun.projects.MlrunProject.enable_model_monitoring`), you effectively deploy three components:
- application controller function: handles the monitoring processing and the triggers the apps that trigger the writer. The controller is a real-time Nuclio job whose frequency is determined by `base_period`. 
- stream function: monitors the log of the data stream. It is triggered when a new log entry is detected. The monitored data is used to create real-time dashboards, detect drift, and analyze performance.
- writer function: writes to the database and outputs alerts.

Model monitoring supports open-source streaming platforms such as Kafka, in addition to integration with the Iguazio AI platform data layer.

## Common terminology
The following terms are used in all the model monitoring pages:
* **Total Variation Distance** (TVD) &mdash; The statistical difference between the actual predictions and the model's trained predictions.
* **Hellinger Distance** &mdash; A type of f-divergence that quantifies the similarity between the actual predictions, and the model's trained predictions.
* **Kullback–Leibler Divergence** (KLD) &mdash; The measure of how the probability distribution of actual predictions is different from the second model's trained reference probability distribution.
* **Model Endpoint** &mdash; A combination of a model and a runtime function that can be a deployed Nuclio function or a job runtime. One function can run multiple endpoints; however, statistics are saved per endpoint.
