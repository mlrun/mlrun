(model-api)=
# Model serving API

MLRun Serving follows the same REST API defined by Triton and [KFServing v2](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md).

Nuclio also supports streaming protocols (Kafka, kinesis, MQTT, etc.). When streaming, the 
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
- **inputs**: List of input elements (numeric values, arrays, or dicts).
- **outputs:** Optional, requested output values.

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


