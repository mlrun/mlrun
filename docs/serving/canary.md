(canary)=
# Canary deployment

Canary deployment is enabled by the API Gateway. The API gateway is a service that exposes your function as a web 
service. Essentially, it is a proxy that forwards requests to your functions and returns the response.
You can configure authentication on the gateway.

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

Use a canary function to test a modified configuration of the Primary function. 
The API traffic is randomly directed to the two functions at the percentages you specify. Start with a low 
percentage for the canary function.  As you see that the canary function works as expected, gradually increase its 
percentage until you turn it into a production function. 

1. Press **Create a canary function** and type in the function name. 
2. Leave the percentages at 5% and 95% to get started, and verify that the canary function works as expected.
2. Gradually increase the percentage, each time verifying its results.
2. When the percentage is high and you are fully satisfied, turn it into a production function by pressing **<img src="../_static/images/kebab-menu.png" width="25"/>**  > **Promote**.
