(model-monitoring)=
# Model monitoring

By definition, ML models in production make inferences on constantly changing data. Even models that have been trained on massive data sets, with the most meticulously labelled data, start to degrade over time, due to concept drift. Changes in the live environment due to changing behavioral patterns, seasonal shifts, new regulatory environments, market volatility, etc., can have a big impact on a trained model’s ability to make accurate predictions.

Model performance monitoring is a basic operational task that is implemented after an AI model has been deployed. Model monitoring includes:

- Built-in model monitoring:
   Machine learning model monitoring is natively built in to the Iguazio MLOps Platform, along with a wide range of 
   model management features and ML monitoring reports. It monitors all of your models in a single, simple, dashboard.

- Automated drift detection:
   Automatically detects concept drift, anomalies, data skew, and model drift in real-time. Even if you are running hundreds of
   models simultaneously, you can be sure to spot and remediate the one that has drifted.

- Automated retraining:
   When drift is detected, Iguazio automatically starts the entire training pipeline to retrain the model, including all relevant 
   steps in the pipeline. The output is a production-ready challenger model, ready to be deployed. This keeps your models up to date, 
   automatically.

- Native feature store integration:
   Feature vectors and labels are stored and analyzed in the Iguazio feature store and are easily compared to the trained 
   features and labels running as part of the model development phase, making it easier for data science teams to 
   collaborate and maintain consistency between AI projects.

See full details and examples in [Model monitoring](../monitoring/index.html).