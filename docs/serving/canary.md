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

- [Create an API gateway](#create-gateway)
- [Create and use a canary function](#canary-function)

<a id="create-gateway"></a>
## Create an API gateway

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
