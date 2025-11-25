(demos)=
# Demos

These end-to-end demos demonstrate how to use the Iguazio AI platform, MLRun, and related tools, to address data science requirements for different industries and implementations.

## Gen AI Demos

|Demo|Description|
|-----------------------------------|---------------------------------------------------------------------------------------------|
|<b>[Call center](https://github.com/mlrun/demo-call-center)</b>|This demo showcases how to use LLMs to turn audio files, from call center conversations between customers and agents, into valuable data &mdash; all in a single workflow orchestrated by MLRun. MLRun automates the entire workflow, auto-scales resources as needed, and automatically logs and ses values between the different workflow steps.|
|<b>[Banking LLM monitoring and feedback loop](https://github.com/mlrun/demo-monitoring-and-feedback-loop/blob/main/README.md)</b>|This demo illustrates how to train, deploy, and monitor, and LLM using an approach described as "LLM as a judge".|
|<b>[Banking Agent Demo](https://github.com/mlrun/demo-banking-agent/blob/main/README.md)|This demo showcases a modular, production-grade banking customer service chatbot. It combines traditional machine learning (churn propensity) and large language models (LLMs) in a single, observable inference pipeline. The system features conditional routing based on guardrails (banking topic and toxicity filtering), and dynamically adapts model behavior using conversation history, sentiment, and churn risk.|

## ML Demos

|Demo|Description|
|-----------------------------------|---------------------------------------------------------------------------------------------|
|<b>[Fraud Prevention (Feature Store)](https://github.com/mlrun/demo-fraud)</b>|This demo shows the usage of MLRun and the feature store. Fraud prevention specifically is a challenge as it requires processing raw transaction and events in real-time and being able to quickly respond and block transactions before they occur. Consider, for example, a case where you would like to evaluate the average transaction amount. When training the model, it is common to take a DataFrame and just calculate the average. However, when dealing with real-time/online scenarios, this average has to be calculated incrementally.|
|<b>[Banking Agent Demo](https://github.com/mlrun/demo-banking-agent/blob/main/README.md)|This demo showcases a modular, production-grade banking customer service chatbot. It combines traditional machine learning (churn propensity) and large language models (LLMs) in a single, observable inference pipeline. The system features conditional routing based on guardrails (banking topic and toxicity filtering), and dynamically adapts model behavior using conversation history, sentiment, and churn risk.|

