(feature-store-available-steps)=
# Feature store's built-in steps

A step runs a function or class handler or a REST API call: MLRun comes with pre-built steps that include data manipulation, readers, writers and model serving.
All steps are supported by the storey engine. 

See full details on built-in steps in {ref}`building-graphs`. 


| Class name                                                  | Description                                                                                                                                                                                                                                                                   |        
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| {py:class}`~mlrun.serving.routers.EnrichmentModelRouter`    | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| {py:class}`~mlrun.serving.routers.EnrichmentVotingEnsemble` | Auto enrich the request with data from the feature store. The router input accepts a list of inference requests (each request can be a dict or a list of incoming features/keys). It enriches the request with data from the specified feature vector (`feature_vector_uri`). |
| {py:class}`~mlrun.feature_store.steps.FeaturesetValidator` | Validate feature values according to the feature set validation policy. Supported also by the Pandas engines. | 

