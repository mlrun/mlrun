(canary)=
# Canary functions and rolling upgrades

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
- [Configuring an API gateway as a canary function using the SDK](#configuring-an-api-gateway-as-a-canary-function-using-the-sdk)
- [Create and use a canary function in the UI](#create-and-use-a-canary-function-in-the-ui)


## Configuring an API gateway as a canary function using the SDK
Assume you have two functions, named fn1 and fn2. 
Define the API gateway entity with canary configuration [60, 40] which means that 60% of traffic goes to fn1 and 40% to fn2.
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

Change the canary function percentages:
```
my_api_gateway.with_canary([fn1, fn2], [10, 90])
```

Update the API gateway:
```
my_api_gateway = project.store_api_gateway(my_api_gateway)
```


<a id="create-gateway"></a>
## Create and use a canary function in the UI

1. Press **Create a canary function** and type in the function name. 
2. Leave the percentages at 5% and 95% to get started, and verify that the canary function works as expected.
2. Gradually increase the percentage, each time verifying its results.
2. When the percentage is high and you are fully satisfied, turn it into a production function by pressing **<img src="../_static/images/kebab-menu.png" width="25"/>**  > **Promote**.

## See also
- {ref}`nuclio-real-time-functions`