(api-handler)=
# API handler

API handlers perform preprocessing on serving graphs invoked by HTTP triggers to add, for example, runtime configuration and administration handlers that perform modifications on the graphs.
API handlers: 
- Support industry-defined API schemas, such as OpenAI interface for LLMs served by the graph
- Implement the v2 KFServing REST API (https://kserve.github.io/website/0.8/modelserving/inference_api/)
- Implement admin APIs for the serving graph, without having to re-deploy them, such as:
  - Enabling and disabling model monitoring on the graph
  - Adding/removing models in real-time

API handlers support Nuclio functions whose trigger is a HTTP trigger, and the mock server.


## Overview
When the GraphServer receives an event with an API handler, and prior to actually sending it to the graph, it evalauates whether the event was sent to a user-configured allowed paths. If yes, it:
- Sends the event to either the root graph step, or another step named in the configuration, as shown in the diagram.
- Executes a method on the event. This is used for administrative operations. Once the admin operation is completed, a response is sent back to the user.
- Can also API handler can perform additional (optional) manipulations on the event body (data), for example, to extract relevant fields from specific paths in the JSON input, and construct the event to be sent to the graph based on the transformations defined by the user.

If the event was sent to an invalid path, it fails the request.


## SDK
The `set_api_handler_config` accepts the full configuration. You cannot add/remove paths and mappings directly on the serving runtime. 





### Guidelines
Path expressions
- Paths are generally assumed to be a specific path, unless globs (*) are used. For example setting a /v1/completion/* path matches any path that starts with `/v1/completion/`, but using `/v1/completion` only meets this specific path.
- If multiple matches exist, the one that is most specific is chosen. For example, if `/v1/completion/*` is set to `allow`, and `/v1/completion/bad` is set to `fail` - calling `/v1/completion/bad` fails the request due to its being more specific.

Body mapping  
- JSONPath may be producing multiple results. For example, in the canonical book store sample, looking for `$['store']['book'][*]['title']` results in all the titles of all the books in the store. The handler should handle this situation and place the results in the field specified as a list of values.
- If the JSONPath search returns a complex object, for example looking for `$['store']['book'][0]` which returns the entire dictionary of the first book (not just the title as above), the results should be placed as-is into the field selected, such that the full result dict is in the field (or a list of these dict results, given previous point).

URL information mapping 
- By default, the API handler adds information to the event about the URL used, and any query params provided. This complements the mapping of the body structure, and allows passing any important information passed in query params or the URL itself. For example, the OpenAI chat-completion API is `GET .../v1/chat/completions/{completion_id}`, then the `completion_id` needs to be available for processing.
- To support that, the API handler parses the URL and splits it to its components, placing it in a field called `mlrun_url_path_segments` (the `mlrun_` denotes a system-provided field). For example, in the URL for chat completion, the parts are [“v1”,”chat”,”completions”,<completion id>]. This allows the graph steps to use this information to extract any information needed.
- In addition, any query params are also placed in an event field called `mlrun_url_query_params`. This is a dictionary mapping from param name to its value(s). For example, the OpenAI get-chat-messages call supports a limit param (among others), so a call to - `GET .../v1/chat/completions/{completion_id}/messages?limit=10` sets the value of this field to {“limit”: 10}. Of course, this field is optional, and its value could be empty if no query params are used.
- This behavior is controlled by the `include_url_info` configuration, which, when set to `False`, bypasses the entire URL parsing logic (URL parts and query params).

Once configured, the API handler configuration is placed in the serving function’s serving spec, so it can be picked up at deployment time and the handler can be instantiated based on it. 

## Viewing API handlers in graphs
In a serving graph visualized in the MLRun UI, the graph displays details on the API handler. For example, you can see allowed paths, and also the transformations between schemas. 
Inn a Jupyter plot of a graph there are indications that an API handler exists without specific details.



## Usage

## Examples