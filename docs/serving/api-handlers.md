(api-handlers)=
API handlers


Add a user-configurable (and some built-in) graph handlers that will be executed by the Graph server and will pre-process any event received by the server before passing it to the graph itself for processing. This pre-process will validate known paths and will be able to re-structure the event body to transform from the incoming event schema to the schema that the processing graph expects. 

Enable users to deploy using MLRun serving graphs while supporting industry-defined API schemas, such as OpenAI interface for LLMs.

Allow users to add runtime configuration and administration handlers that will be able to perform modifications on the graph without having to re-deploy it.





Supporting the OpenAI interface for LLMs served by the graph.

Implement the v2 KFServing REST API (https://kserve.github.io/website/0.8/modelserving/inference_api/) - this is needed to replace the current router topology.

Implement admin APIs for the serving graph, such as:

Enabling and disabling MM on the graph.

Adding/removing models in real-time.

Supports 
- Nuclio functions only when the trigger is a HTTP trigger.
- Mock server  

## SDK
The `set_api_handler_config` accepts the full configuration, there will be no granular support of adding/removing paths and mappings directly on the serving runtime. The ApiHandlerConfig class will enable doing this as mentioned in the api above.

Path expressions
- Paths are generally assumed to be a specific path, unless globs (*) are used. For example setting a /v1/completion/* path will match any path that starts with /v1/completion/, but using /v1/completion will only meet this specific path.
- If multiple matches exist, the one that is most specific will be chosen. For example, if /v1/completion/* is set to allow, and /v1/completion/bad is set to fail - calling /v1/completion/bad will fail the request due to it being more specific.

Body mapping  
- JSONPath may be producing multiple results, for example in the canonical book store sample, looking for $['store']['book'][*]['title'] will result in all the titles of all the books in the store. The handler should handle this situation and place the results in the field specified as a list of values.
- If the JSONPath search returns a complex object, for example looking for $['store']['book'][0] which will return the entire dictionary of the first book (not just the title as above), the results should be placed as-is into the field selected. Meaning that the full result dict will be in the field (or a list of these dict results, given previous point).

URL information mapping 
- By default, the API handler will add information to the event about the URL used, and any query params provided. This is to complement the mapping of the body structure, and to allow passing any important information passed in query params or the URL itself. For example - the OpenAI chat-completion API is GET .../v1/chat/completions/{completion_id} - the completion_id needs to be available for processing.
- To support that, the API handler will parse the URL and split it to its components, placing it in a field called mlrun_url_path_segments (the mlrun_ is meant to denote a system-provided field). For example, in the URL for chat completion, the parts will be [“v1”,”chat”,”completions”,<completion id>]. This will allow the graph steps to use this information to extract any information needed.
- In addition, any query params will be also placed in an event field called mlrun_url_query_params. This will be a dictionary mapping from param name to its value(s). For example, the OpenAI get-chat-messages call supports a limit param (among others), so a call to - GET .../v1/chat/completions/{completion_id}/messages?limit=10 will set the value of this field to {“limit”: 10}. Of course, this field is optional, and its value may be empty if no query params are used.
- This behavior will be controller by the include_url_info configuration, which when set to False will bypass the entire URL parsing logic (URL parts and query params).

Once configured, the API handler configuration is placed in the serving function’s serving spec, so it can be picked up at deployment time and the handler can be instantiated based on it. 