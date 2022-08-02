(testing-model-server)=
# Testing a model server

MLRun provides a mock server as part of the `serving` runtime. This gives you the ability to deploy your serving function in your local environment for testing purposes.

```python
serving_fn = code_to_function(name='myService', kind='serving', image='mlrun/mlrun')
serving_fn.add_model('my_model', model_path=model_file_path)
server = serving_fn.to_mock_server()
```

You can use test data and programmatically invoke the `predict()` method of mock server. In this example, the model is expecting a python dictionary as input.

```python
my_data = '''{"inputs":[[5.1, 3.5, 1.4, 0.2],[7.7, 3.8, 6.7, 2.2]]}'''
server.test("/v2/models/my_model/infer", body=my_data)
```

Output:
2022-03-29 09:44:52,687 [info] model my_model was loaded
2022-03-29 09:44:52,688 [info] Loaded ['my_model']

    {'id': '0282c63bff0a44cabfb9f06c34489035',
    'model_name': 'my_model',
    'outputs': [0, 2]}


The data structure used in the body parameter depends on how the `predict()` method of the model server is defined. For examples of how to define your own model server class, see [here](custom-model-serving-class.html#predict-method).

To review the mock server api, see [here](../api/mlrun.runtimes.html#mlrun.runtimes.ServingRuntime.to_mock_server).