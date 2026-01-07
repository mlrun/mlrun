.. _mlrun.serving:

mlrun.serving
==============

.. autoclass:: mlrun.serving.states.BaseStep
   :members: to, error_handler, set_flow, cycle_to
   :private-members:

.. autoclass:: mlrun.serving.states.RootFlowStep
   :members: add_step, add_shared_model, configure_shared_pool_resource
   :private-members:

.. autoclass:: mlrun.serving.states.ModelRunnerStep
   :members: add_model, add_shared_model_proxy, configure_pool_resource
   :private-members:

.. autoclass:: mlrun.serving.states.Model
   :members: predict, predict_async, load
   :private-members:

.. autoclass:: mlrun.serving.states.LLModel
   :no-members:

.. autoclass:: mlrun.serving.states.ModelRunnerSelector
    :members: select_models, select_outlets
    :private-members:

.. automodule:: mlrun.serving
   :members:
   :show-inheritance:
   :undoc-members:
   :exclude-members: LLModel, Model, ModelRunnerStep, ModelRunner, ModelSelector, MonitoredStep, ModelRunnerSelector

.. automodule:: mlrun.serving.remote
   :members:
   :undoc-members:

.. automodule:: mlrun.serving.routers
   :members:
   :undoc-members:

.. autoclass:: mlrun.serving.utils.StepToDict
   :members:

.. autoclass:: mlrun.serving.states.MonitoredStep
   :members:
   :private-members: _calculate_monitoring_data