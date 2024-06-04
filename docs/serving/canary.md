(canary)=
# Canary and rolling upgrades

```{admonition} Note
Relevant when MLRun is executed in the [Iguazio platform](https://www.iguazio.com/docs/latest-release/) (**"the platform"**).
```

Canary rollout is a known practice to first test a software update on a small number of users before rolling it 
out to all users. In machine learning, the main usage is to test a new model on a small subset of users before 
rolling it out to all users. 

Canary functions are defined using an API gateway. The API gateway is a service that exposes your function as a 
web service. Essentially, it is a proxy that forwards requests to your functions and returns the response.
You can configure authentication on the gateway.

The API traffic is randomly directed to the two functions at the percentages you specify. Start with a low 
percentage for the canary function. Verify that the canary function works as expected (or modify it until it does 
work as desired). Then gradually increase its percentage until you turn it into a production function. 

**In this section**
- [Creating and managing API gateways using the SDK](#creating-and-managing-api-gateways-using-the-sdk)
- [Create an API gateway in the UI](#create-gateway)
- [Create and use a canary function](#canary-function)

## Creating and managing API gateway using the SDK

### Create an API gateway with basic authorization

Assume you already have a deployed nuclio function:
```
fn = mlrun.code_to_function(
            filename="nuclio_function.py",
            name=f"nuclio-func",
            kind="nuclio",
            image="python:3.9",
            handler="handler",
        )
```		

Create the gateway with basic auth:
```
# Define the API gateway entity
my_api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
            mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
                name="gw-with-basic-auth",
            ),
            mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
                functions=fn,
                project=project.name,
            ),
    )
# add basic authorization configuration
my_api_gateway.with_basic_auth(username="test",
 password="pass")

# create (or update) the API gateway
# It's crucial to update the defined API Gateway entity with the entity returned from store_api_gateway method. Thia fills in important fields like host/path, etc.

my_api_gateway = project.store_api_gateway(my_api_gateway)

# even though store_api_gateway checks that api_gateway is ready to be invoked before returning it, there is a way to check if api gateway is in ready state
my_api_gateway.is_ready()

# If you know that api gateway was changed (for example, in the UI), you can easily load the latest changes:
my_api_gateway.sync()
```
### Invoke the API gateway
Since the gateway is configured with basic auth, you need to pass authorization credentials. 
```
response = my_api_gateway.invoke(auth=("test", "pass"), verify=False)
```
### Using the API Gateway as a Canary function
Assume you have two functions defined as above, named fn1 and fn2. 
Define the API gateway entity with canary configuration [60, 40] which means that 60% of traffic will go to function1 and 40% to function2 
```
my_api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
            name="gw-canary",
        ),
        mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            functions=[fn1, fn2],
            project=project.name,
            canary=[60, 40],
        ),
    )
```

Create (or update) the API gateway:
```
my_api_gateway = project.store_api_gateway(my_api_gateway)
```


<a id="create-gateway"></a>
## Create an API gateway in the UI

To create an API gateway in the UI:
1. In your project page, press **API Gateways** tab, then press **NEW API GATEWAY**.
2. Select an **Authentication Mode**:
   - None (default)
   - Basic
   - Access key
   - OAuth2
   
   and fill in any required values.
2. Type in the API Gateway parameters:
   - **Name**: The name of the API Gateway. Required
   - **Description**: A description of the API Gateway.
   - **Host**: The host of the API Gateway. (Relevant for open-source only.)
   - **Path**: The path of the API Gateway.
2. In **Primary**, type in the function that is triggered via the API Gateway. 

## Create and use a canary function

1. Press **Create a canary function** and type in the function name. 
2. Leave the percentages at 5% and 95% to get started, and verify that the canary function works as expected.
2. Gradually increase the percentage, each time verifying its results.
2. When the percentage is high and you are fully satisfied, turn it into a production function by pressing **<img src="../_static/images/kebab-menu.png" width="25"/>**  > **Promote**.
