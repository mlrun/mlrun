(feature-store)=
# Feature store 

The feature store is a centralized and versioned catalog where everyone can engineer and store features along with their metadata and statistics, share them and reuse them, and analyze their impact on existing models. The feature store plugs seamlessly into the data ingestion, model training, model serving, and model monitoring components, eliminating significant development and operations overhead, and delivering exceptional performance. Users can simply group together independent features into vectors, and use those from their jobs or real-time services. Iguazio’s high performance engines take care of automatically joining and accurately computing the features.<br>
You can use the feature store throughout the MLOps flow:
1. {ref}`Ingesting data <feature-store-data-ingestion>`
2. {ref}`Training <model-training>`
2. {ref}`Model serving <model_serving>`

See also the feature store tutorial {ref}`basic-demo`.

**In this section**

```{toctree}
:maxdepth: 1

fs-overview
fsets-ingest-transform
feature-vectors
./end-to-end-demo/index
```





