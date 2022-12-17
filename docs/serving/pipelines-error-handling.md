(pipelines-error-handling)=
# Error handling
   
Graph steps might raise an exception. If you want to have an error handling flow, you can specify an exception handling 
step/branch that is triggered on error. The error handler step receives the event that entered the failed step, 
with two extra attributes: `event.origin_state` indicates the name of the failed step; and `event.error` holds the error string.

Use the `graph.error_handler()` (apply to all steps) or `step.error_handler()` (apply to a specific step) 
if you want the error from the graph or the step to be fed into a specific step (catcher).

Example of setting an error catcher per step:
```
graph.add_step("MyClass", name="my-class", after="pre-process").error_handler("catcher")
graph.add_step("ErrHandler", name="catcher", full_event=True, after="")
```
```{admonition} Note
Additional steps can follow the catcher step.
```
Using the example in [Model serving graph](./model-serving-get-started.html#flow), you can add an error handler as follows:
```
graph2_enrich.error_handler("catcher")
graph2.add_step("ErrHandler", name="catcher", full_event=True, after="")
```
```
<mlrun.serving.states.TaskStep at 0x7fd46e557750>
```

Now, display the graph again:
```
graph2.plot(rankdir='LR')
```
```
<mlrun.serving.states.TaskStep at 0x7fd46e557750>
```

## Exception stream
The graph errors/exceptions can be pushed into a special error stream. This is very convenient in the case of 
distributed and production graphs.

To set the exception stream address (using v3io streams uri):
```
fn_preprocess2.spec.error_stream = err_stream
```