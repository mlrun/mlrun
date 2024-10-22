(demos)=
# Demos

These end-to-end demos demonstrate how to use the Iguazio AI platform, MLRun, and related tools, to address data science requirements for different industries and implementations.

## Gen AI Demos

|Demo|Description|
|-----------------------------------|---------------------------------------------------------------------------------------------|
|<b>[Call center demo](https://github.com/mlrun/demo-call-center)</b>|This demo showcases how to use LLMs to turn audio files, from call center conversations between customers and agents, into valuable data &mdash; all in a single workflow orchestrated by MLRun. MLRun automates the entire workflow, auto-scales resources as needed, and automatically logs and ses values between the different workflow steps.|
|<b>[Fine tune an LLM and build a BOT](https://github.com/mlrun/demo-llm-tuning/blob/main)</b>|This demo shows how to fine-tune an LLM and build a chatbot that can answer all your questions about MLRun's MLOps. It starts with a pre-trained model from Hugging Face, fine tunes the model, creates an automated training pipeline, and deploys a serving graph. The serving graph includes post-processing for accuracy of generated text, and filtering for toxicity.|
|<b>[Interactive bot demo using LLMs](https://github.com/mlrun/demo-llm-bot/blob/main/README.md)</b>|This demo showcases the usage of Language Models (LLMs) and MLRun to build an interactive chatbot using your own data for Retrieval Augmented Question Answering. The data will be ingested and indexed into a Vector Database to be queried by an LLM in real-time. The project utilizes MLRun for orchestration/deployment, HuggingFace embeddings for indexing data, Milvus for the vector database, OpenAI's GPT-3.5 model for generating responses, Langchain to retrieve relevant data from the vector store and augment the response from the LLM, and Gradio for building an interactive frontend.|


## ML Demos

|Demo|Description|
|-----------------------------------|---------------------------------------------------------------------------------------------|
|<b>[Mask Detection Demo](https://github.com/mlrun/demo-mask-detection)</b>|This demo contains three notebooks that: Serve the model as a serverless function in an http endpoint; Train and evaluate a model for detecting if an image includes a person who is wearing a mask, by using Tensorflow, Keras, or PyTorch; Write an automatic pipeline where you download a dataset of images, train and evaluate the model, then optimize the model (using ONNX) and serve it.|
|<b>[Fraud Prevention (Feature Store)](https://github.com/mlrun/demo-fraud)</b>|This demo shows the usage of MLRun and the feature store. Fraud prevention specifically is a challenge as it requires processing raw transaction and events in real-time and being able to quickly respond and block transactions before they occur. Consider, for example, a case where you would like to evaluate the average transaction amount. When training the model, it is common to take a DataFrame and just calculate the average. However, when dealing with real-time/online scenarios, this average has to be calculated incrementally.|
|<b>[Sagemaker demo](https://github.com/mlrun/demo-sagemaker)</b>|This demo showcases how to build, manage, and deploy ML models using AWS SageMaker and MLRun. It emphasizes the automation of ML workflows from development to production.|
|<b>[Building Production Pipelines WIth AzureML and MLRun](https://github.com/mlrun/demos/tree/1.7.x/azureml-demo) </b>|This demo uses the MLRun Feature Store to ingest and prepare data, create an offline feature vector (snapshot) for training, run AzureML AutoML Service as an automated step (function) in MLRun, view and compare the AzureML Models using MLRun tools, Build a real-time serving pipeline, and provide real-time model monitoring. |