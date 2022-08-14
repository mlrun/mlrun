# API Gateway

API Gateway is a service that allows you to expose your functions as a web service.
Essentially, it is a proxy that forwards requests to your functions and returns the response.
It can be used to invoke your functions, and can provide authentication, canary deployments, and other features.

**In this section**

- [Create an API gateway](#create-gateway)
- [Canary function](#canary-function)

<a id="create-gateway"></a>
## Create an API gateway

To create an API gateway in the UI:
1. In your project page, press **API Gateways** tab, then press **NEW API Gateway**.
2. Select an **Authentication Mode**:
   - None
   - Basic
   - Access key
   - OAuth2
2. Type in the API Gateway parameters:
   - **Name**: The name of the API Gateway. Required
   - **Description**: A description of the API Gateway.
   - **Host**: The host of the API Gateway. (Relevant for open-source only.)
   - **Path**: The path of the API Gateway.
2. In **Primary**, type in the function that is triggered via the API Gateway. 

## Canary function

You can add a canary function to the API gateway, for testing purposes. You control the percentage of traffic that goes to a canary function by changing the percentage of the upstream. As you see that the canary function works well, gradually increase its percentage. When you are satisfied with its performance, turn it into a production function, and remove it as a canary function.

1. Press **Create a canary function** and type in the function name. 
2. Adjust the percentage of traffic that is sent to the canary function.