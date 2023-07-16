(pipelines-error-handling)=
# Error handling
   
Graph steps might raise an exception. You can define exception handling (an error handling flow) that is triggered on error. The exception can be on a:
* step: The error handler is appended to the step that, if it fails, triggers the error handling. If you want 
the graph to continue after an error handler execution, specify the next step in the `before` parameter. 
If you want the graph to complete after an error handler execution, omit the `before` parameter.
* graph: When set on the graph object, the graph completes after the error handler execution.


Example of an exception on a step that only runs when/if the "pre-process" step fails:
```
graph = function.set_topology('flow', engine='async')
graph.to(name='pre-process', handler='raising_step').error_handler(name='catcher', handler='handle_error', full_event=True, before='echo')

# Add another step after pre-process step or the error handling
graph.add_step(name="echo", handler='echo', after="pre-process").respond()
graph
```

Example of an exception on a graph:
```
graph = function.set_topology('flow', engine='async')
graph.error_handler(name='error_catcher', handler='handle_error', full_event=True, before='echo')
graph.to(name='raise', handler='raising_step').to(name="echo", handler='echo', after="raise").respond()
```

See full parameter description in {py:class}`~mlrun.serving.states.BaseStep.error_handler`.


## Exception stream
    
The graph errors/exceptions can be pushed into a special error stream. This is very convenient in the case of 
distributed and production graphs.

To set the exception stream address (using v3io streams uri):
```
fn_preprocess2.spec.error_stream = err_stream
```