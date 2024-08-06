(model-api)=
# Model serving API

MLRun Serving follows the same REST API defined by Triton and [KFServing v2](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md).

Nuclio also supports streaming protocols (Kafka, Kinesis, MQTT, etc.). When streaming, the 
`model` name and `operation` can be encoded inside the message body.

The APIs are:
* [explain](#explain)
* [get model health / readiness](#get-model-health-readiness)
* [get model metadata](#get-model-metadata)
* [get server info](#get-server-info)
* [infer / predict](#infer-predict)
* [list models](#list-models)

## explain

POST /v2/models/<model>[/versions/{VERSION}]/explain

Request body:

    {
      "id" : $string #optional,
      "model" : $string #optional
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

Response structure:

    {
      "model_name" : $string,
      "model_version" : $string #optional,
      "id" : $string,
      "outputs" : [ $response_output, ... ]
    }
    
## get model health / readiness

    GET v2/models/${MODEL_NAME}[/versions/${VERSION}]/ready

Returns 200 for Ok, 40X for not ready.


## get model metadata

    GET v2/models/${MODEL_NAME}[/versions/${VERSION}]

Response example: `{"name": "m3", "version": "v2", "inputs": [..], "outputs": [..]}`

## get server info

    GET /
    GET /v2/health

Response example: `{'name': 'my-server', 'version': 'v2', 'extensions': []}`

## infer / predict

    POST /v2/models/<model>[/versions/{VERSION}]/infer

Request body:

    {
      "id" : $string #optional,
      "model" : $string #optional
      "data_url" : $string #optional
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

- **id**: Unique Id of the request, if not provided a random value is provided.
- **model**: Model to select (for streaming protocols without URLs).
- **data_url**: Option to load the `inputs` from an external file/s3/v3io/.. object.
- **parameters**: Optional request parameters.
- **inputs**: Inputs for a model, where each data point should be provided as a list. 
Each data point can be extracted from different features with varying types, the feature have to be serializable.
  1. **Single Data Point Input:** 
     - Accepts a list representing a single data point, which can include features of different types.
     - Example: `[feature1, feature2, feature3, ...]`
  2. **Batch Input:**
     - Allows a list of lists for processing multiple data points simultaneously, 
with each data point containing features of different types.
     - Example: `[[feature1a, feature2a, feature3a, ...], [feature1b, feature2b, feature3b, ...], ..]`

  - Note: If the user wants to send an image as an input, it should be sent as a list of RGB lists. 
  This format is only accepted in the **batch** case. For example: `[[[[R1, G1, B1], [R2, G2, B2], ...]],...]`

- **outputs:** Optional, requested output values.

## infer_dict / predict_dict

    POST /v2/models/<model>[/versions/{VERSION}]/infer_dict

Request body:

    {
      "id" : $string #optional,
      "model" : $string #optional
      "data_url" : $string #optional
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

- **id**: Unique Id of the request, if not provided a random value is provided.
- **model**: Model to select (for streaming protocols without URLs).
- **data_url**: Option to load the `inputs` from an external file/s3/v3io/.. object.
- **parameters**: Optional request parameters.
- **inputs**: Inputs for a model, where each data point should be provided as a dictionary. 
Each data point can be extracted from different features with varying types, the feature have to be serializable. 
This API supports only batch mode.
1**Batch Input:**
     - Allows a list of dictionaries for processing multiple data points simultaneously, with each data point containing features of different types.
     - Example: `[{feature1a: value1a, feature2a: value2a, feature3a: value3a, ...} ..]`
- **outputs:** Optional, requested output values.

### Additional Information:
- This API is particularly helpful when the user does not remember the order of the features.
- The API can be used only if the model was logged with a schema.
- When using this API, the predict method of the model server will still receive a 
list of lists with the features in the correct order.

```{note} You can also send binary data to the function, for example, a JPEG image. The serving engine pre-processor 
detects it based on the HTTP content-type and converts it to the above request structure, placing the 
image bytes array in the `inputs` field.
```
    
Response structure:

    {
      "model_name" : $string,
      "model_version" : $string #optional,
      "id" : $string,
      "outputs" : [ $response_output, ... ]
    }

## list models

    GET /v2/models/

Response example:  `{"models": ["m1", "m2", "m3:v1", "m3:v2"]}`


