# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "TaskStep",
    "RouterStep",
    "RootFlowStep",
    "ErrorStep",
    "MonitoringApplicationStep",
]

import os
import pathlib
import traceback
from copy import copy, deepcopy
from inspect import getfullargspec, signature
from typing import Any, Optional, Union, cast

import storey.utils

import mlrun
import mlrun.common.schemas as schemas
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfileKafkaTarget,
    DatastoreProfileV3io,
    datastore_profile_read,
)
from mlrun.datastore.storeytargets import KafkaStoreyTarget, StreamStoreyTarget
from mlrun.utils import logger

from ..config import config
from ..datastore import get_stream_pusher
from ..datastore.utils import (
    get_kafka_brokers_from_dict,
    parse_kafka_url,
)
from ..errors import MLRunInvalidArgumentError, err_to_str
from ..model import ModelObj, ObjectDict
from ..platforms.iguazio import parse_path
from ..utils import get_class, get_function, is_explicit_ack_supported
from .utils import StepToDict, _extract_input_data, _update_result_body

callable_prefix = "_"
path_splitter = "/"
previous_step = "$prev"
queue_class_names = [">>", "$queue"]

MAX_MODELS_PER_ROUTER = 5000


class GraphError(Exception):
    """error in graph topology or configuration"""

    pass


class StepKinds:
    router = "router"
    task = "task"
    flow = "flow"
    queue = "queue"
    choice = "choice"
    root = "root"
    error_step = "error_step"
    monitoring_application = "monitoring_application"


_task_step_fields = [
    "kind",
    "class_name",
    "class_args",
    "handler",
    "skip_context",
    "after",
    "function",
    "comment",
    "shape",
    "full_event",
    "on_error",
    "responder",
    "input_path",
    "result_path",
    "model_endpoint_creation_strategy",
    "endpoint_type",
]

_default_fields_to_strip_from_step = [
    "model_endpoint_creation_strategy",
    "endpoint_type",
]


def new_remote_endpoint(
    url: str,
    creation_strategy: schemas.ModelEndpointCreationStrategy,
    endpoint_type: schemas.EndpointType,
    **class_args,
):
    class_args = deepcopy(class_args)
    class_args["url"] = url
    return TaskStep(
        "$remote",
        class_args=class_args,
        model_endpoint_creation_strategy=creation_strategy,
        endpoint_type=endpoint_type,
    )


class BaseStep(ModelObj):
    kind = "BaseStep"
    default_shape = "ellipse"
    _dict_fields = ["kind", "comment", "after", "on_error"]
    _default_fields_to_strip = _default_fields_to_strip_from_step

    def __init__(
        self,
        name: Optional[str] = None,
        after: Optional[list] = None,
        shape: Optional[str] = None,
    ):
        self.name = name
        self._parent = None
        self.comment = None
        self.context = None
        self.after = after or []
        self._next = None
        self.shape = shape
        self.on_error = None
        self._on_error_handler = None
        self.model_endpoint_creation_strategy = (
            schemas.ModelEndpointCreationStrategy.SKIP
        )

    def get_shape(self):
        """graphviz shape"""
        return self.shape or self.default_shape

    def set_parent(self, parent):
        """set/link the step parent (flow/router)"""
        self._parent = parent

    @property
    def next(self):
        return self._next

    @property
    def parent(self):
        """step parent (flow/router)"""
        return self._parent

    def set_next(self, key: str):
        """set/insert the key as next after this step, optionally remove other keys"""
        if not self.next:
            self._next = [key]
        elif key not in self.next:
            self._next.append(key)
        return self

    def after_step(self, *after, append=True):
        """specify the previous step names"""
        # add new steps to the after list
        if not append:
            self.after = []
        for name in after:
            # if its a step/task class (vs a str) extract its name
            name = name if isinstance(name, str) else name.name
            if name not in self.after:
                self.after.append(name)
        return self

    def error_handler(
        self,
        name: Optional[str] = None,
        class_name=None,
        handler=None,
        before=None,
        function=None,
        full_event: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        **class_args,
    ):
        """set error handler on a step or the entire graph (to be executed on failure/raise)

        When setting the error_handler on the graph object, the graph completes after the error handler execution.

        example:
            in the below example, an 'error_catcher' step is set as the error_handler of the 'raise' step:
            in case of error/raise in 'raise' step, the handle_error will be run. after that,
            the 'echo' step will be run.
            graph = function.set_topology('flow', engine='async')
            graph.to(name='raise', handler='raising_step')\
                .error_handler(name='error_catcher', handler='handle_error', full_event=True, before='echo')
            graph.add_step(name="echo", handler='echo', after="raise").respond()

        :param name:        unique name (and path) for the error handler step, default is class name
        :param class_name:  class name or step object to build the step from
                            the error handler step is derived from task step (ie no router/queue functionally)
        :param handler:     class/function handler to invoke on run/event
        :param before:      string or list of next step(s) names that will run after this step.
                            the `before` param must not specify upstream steps as it will cause a loop.
                            if `before` is not specified, the graph will complete after the error handler execution.
        :param function:    function this step should run in
        :param full_event:  this step accepts the full event (not just the body)
        :param input_path:  selects the key/path in the event to use as input to the step
                            this requires that the event body will behave like a dict, for example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means the step will
                            receive 7 as input
        :param result_path: selects the key/path in the event to write the results to
                            this requires that the event body will behave like a dict, for example:
                            event: {"x": 5} , result_path="y" means the output of the step will be written
                            to event["y"] resulting in {"x": 5, "y": <result>}
        :param class_args:  class init arguments

        """
        if not (class_name or handler):
            raise MLRunInvalidArgumentError("class_name or handler must be provided")
        if isinstance(self, RootFlowStep) and before:
            raise MLRunInvalidArgumentError(
                "`before` arg can't be specified for graph error handler"
            )

        name = get_name(name, class_name)
        step = ErrorStep(
            class_name,
            class_args,
            handler,
            name=name,
            function=function,
            full_event=full_event,
            input_path=input_path,
            result_path=result_path,
        )
        self.on_error = name
        before = [before] if isinstance(before, str) else before
        step.before = before or []
        step.base_step = self.name
        if hasattr(self, "_parent") and self._parent:
            # when self is a step
            step = self._parent._steps.update(name, step)
            step.set_parent(self._parent)
        else:
            # when self is the graph
            step = self._steps.update(name, step)
            step.set_parent(self)

        return self

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        """init the step class"""
        self.context = context

    def _is_local_function(self, context):
        return True

    def get_children(self):
        """get child steps (for router/flow)"""
        return []

    def __iter__(self):
        yield from []

    @property
    def fullname(self):
        """full path/name (include parents)"""
        name = self.name or ""
        if self._parent and self._parent.fullname:
            name = path_splitter.join([self._parent.fullname, name])
        return name.replace(":", "_")  # replace for graphviz escaping

    def _post_init(self, mode="sync"):
        pass

    def _set_error_handler(self):
        """init/link the error handler for this step"""
        if self.on_error:
            error_step = self.context.root.path_to_step(self.on_error)
            self._on_error_handler = error_step.run

    def _log_error(self, event, err, **kwargs):
        """on failure log (for sync mode)"""
        error_message = err_to_str(err)
        self.context.logger.error(
            f"step {self.name} got error {error_message} when processing an event:\n {event.body}"
        )
        error_trace = traceback.format_exc()
        self.context.logger.error(error_trace)
        self.context.push_error(
            event, f"{error_message}\n{error_trace}", source=self.fullname, **kwargs
        )

    def _call_error_handler(self, event, err, **kwargs):
        """call the error handler if exist"""
        if not event.error:
            event.error = {}
        event.error[self.name] = err_to_str(err)
        event.origin_state = self.fullname
        return self._on_error_handler(event)

    def path_to_step(self, path: str):
        """return step object from step relative/fullname"""
        path = path or ""
        tree = path.split(path_splitter)
        next_level = self
        for step in tree:
            if step not in next_level:
                raise GraphError(
                    f"step {step} doesnt exist in the graph under {next_level.fullname}"
                )
            next_level = next_level[step]
        return next_level

    def to(
        self,
        class_name: Union[str, StepToDict] = None,
        name: Optional[str] = None,
        handler: Optional[str] = None,
        graph_shape: Optional[str] = None,
        function: Optional[str] = None,
        full_event: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_endpoint_creation_strategy: Optional[
            schemas.ModelEndpointCreationStrategy
        ] = None,
        **class_args,
    ):
        """add a step right after this step and return the new step

        example:
            a 4-step pipeline ending with a stream:
            graph.to('URLDownloader')\
                 .to('ToParagraphs')\
                 .to(name='to_json', handler='json.dumps')\
                 .to('>>', 'to_v3io', path=stream_path)\

        :param class_name:  class name or step object to build the step from
                            for router steps the class name should start with '*'
                            for queue/stream step the class should be '>>' or '$queue'
        :param name:        unique name (and path) for the child step, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param graph_shape: graphviz shape name
        :param function:    function this step should run in
        :param full_event:  this step accepts the full event (not just body)
        :param input_path:  selects the key/path in the event to use as input to the step
                            this requires that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means the step will
                            receive 7 as input
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="y" means the output of the step will be written
                            to event["y"] resulting in {"x": 5, "y": <result>}
        :param model_endpoint_creation_strategy: Strategy for creating or updating the model endpoint:

                            * **overwrite**:

                            1. If model endpoints with the same name exist, delete the `latest` one.
                            2. Create a new model endpoint entry and set it as `latest`.

                            * **inplace** (default):

                            1. If model endpoints with the same name exist, update the `latest` entry.
                            2. Otherwise, create a new entry.

                            * **archive**:

                            1. If model endpoints with the same name exist, preserve them.
                            2. Create a new model endpoint with the same name and set it to `latest`.

        :param class_args:  class init arguments
        """
        if hasattr(self, "steps"):
            parent = self
        elif self._parent:
            parent = self._parent
        else:
            raise GraphError(
                f"step {self.name} parent is not set or it's not part of a graph"
            )

        name, step = params_to_step(
            class_name,
            name,
            handler,
            graph_shape=graph_shape,
            function=function,
            full_event=full_event,
            input_path=input_path,
            result_path=result_path,
            class_args=class_args,
            model_endpoint_creation_strategy=model_endpoint_creation_strategy,
        )
        step = parent._steps.update(name, step)
        step.set_parent(parent)
        if not hasattr(self, "steps"):
            # check that its not the root, todo: in future may gave nested flows
            step.after_step(self.name)
        parent._last_added = step
        return step

    def set_flow(
        self,
        steps: list[Union[str, StepToDict, dict[str, Any]]],
        force: bool = False,
    ):
        """
        Set list of steps as downstream from this step, in the order specified. This will overwrite any existing
        downstream steps.

        :param steps: list of steps to follow this one
        :param force: whether to overwrite existing downstream steps. If False, this method will fail if any downstream
                      steps have already been defined. Defaults to False.

        :return: the last step added to the flow

        example::

            The below code sets the downstream nodes of step1 by using a list of steps (provided to `set_flow()`) and a
            single step (provided to `to()`), resulting in the graph (step1 -> step2 -> step3 -> step4).
            Notice that using `force=True` is required in case step1 already had downstream nodes (e.g. if the existing
            graph is step1 -> step2_old) and that following the execution of this code the existing downstream steps
            are removed. If the intention is to split the graph (and not to overwrite), please use `to()`.

            step1.set_flow(
                [
                    dict(name="step2", handler="step2_handler"),
                    dict(name="step3", class_name="Step3Class"),
                ],
                force=True,
            ).to(dict(name="step4", class_name="Step4Class"))
        """
        raise NotImplementedError("set_flow() can only be called on a FlowStep")

    def supports_termination(self):
        return False


class TaskStep(BaseStep):
    """task execution step, runs a class or handler"""

    kind = "task"
    _dict_fields = _task_step_fields
    _default_class = ""

    def __init__(
        self,
        class_name: Optional[Union[str, type]] = None,
        class_args: Optional[dict] = None,
        handler: Optional[str] = None,
        name: Optional[str] = None,
        after: Optional[list] = None,
        full_event: Optional[bool] = None,
        function: Optional[str] = None,
        responder: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_endpoint_creation_strategy: Optional[
            schemas.ModelEndpointCreationStrategy
        ] = schemas.ModelEndpointCreationStrategy.SKIP,
        endpoint_type: Optional[schemas.EndpointType] = schemas.EndpointType.NODE_EP,
    ):
        super().__init__(name, after)
        self.class_name = class_name
        self.class_args = class_args or {}
        self.handler = handler
        self.function = function
        self._handler = None
        self._object = None
        self._async_object = None
        self.skip_context = None
        self.context = None
        self._class_object = None
        self.responder = responder
        self.full_event = full_event
        self.input_path = input_path
        self.result_path = result_path
        self.on_error = None
        self._inject_context = False
        self._call_with_event = False
        self.model_endpoint_creation_strategy = model_endpoint_creation_strategy
        self.endpoint_type = endpoint_type

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
        self._async_object = None
        if not self._is_local_function(context):
            # skip init of non local functions
            return

        if self.handler and not self.class_name:
            # link to function
            if callable(self.handler):
                self._handler = self.handler
                self.handler = self.handler.__name__
            else:
                self._handler = get_function(self.handler, namespace)
            args = signature(self._handler).parameters
            if args and "context" in list(args.keys()):
                self._inject_context = True
            self._set_error_handler()
            return

        self._class_object, self.class_name = self.get_step_class_object(
            namespace=namespace
        )
        if not self._object or reset:
            # init the step class + args
            extracted_class_args = self.get_full_class_args(
                namespace=namespace,
                class_object=self._class_object,
                **extra_kwargs,
            )
            try:
                self._object = self._class_object(**extracted_class_args)
            except TypeError as exc:
                raise TypeError(
                    f"failed to init step {self.name}\n args={self.class_args}"
                ) from exc

            # determine the right class handler to use
            handler = self.handler
            if handler:
                if not hasattr(self._object, handler):
                    raise GraphError(
                        f"handler ({handler}) specified but doesnt exist in class {self.class_name}"
                    )
            else:
                if hasattr(self._object, "do_event"):
                    handler = "do_event"
                    self._call_with_event = True
                elif hasattr(self._object, "do"):
                    handler = "do"
            if handler:
                self._handler = getattr(self._object, handler, None)

        self._set_error_handler()
        if mode != "skip":
            self._post_init(mode)

    def get_full_class_args(self, namespace, class_object, **extra_kwargs):
        class_args = {}
        for key, arg in self.class_args.items():
            if key.startswith(callable_prefix):
                class_args[key[1:]] = get_function(arg, namespace)
            else:
                class_args[key] = arg
        class_args.update(extra_kwargs)

        if not isinstance(self, MonitoringApplicationStep):
            # add common args (name, context, ..) only if target class can accept them
            argspec = getfullargspec(class_object)

            for key in ["name", "context", "input_path", "result_path", "full_event"]:
                if argspec.varkw or key in argspec.args:
                    class_args[key] = getattr(self, key)
            if argspec.varkw or "graph_step" in argspec.args:
                class_args["graph_step"] = self
        return class_args

    def get_step_class_object(self, namespace):
        class_name = self.class_name
        class_object = self._class_object
        if isinstance(class_name, type):
            class_object = class_name
            class_name = class_name.__name__
        elif not class_object:
            if class_name == "$remote":
                from mlrun.serving.remote import RemoteStep

                class_object = RemoteStep
            else:
                class_object = get_class(class_name or self._default_class, namespace)
        return class_object, class_name

    def _is_local_function(self, context):
        # detect if the class is local (and should be initialized)
        current_function = get_current_function(context)
        if current_function == "*":
            return True
        if not self.function and not current_function:
            return True
        if (
            self.function and self.function == "*"
        ) or self.function == current_function:
            return True
        return False

    @property
    def async_object(self):
        """return the sync or async (storey) class instance"""
        return self._async_object or self._object

    def clear_object(self):
        self._object = None

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(
                mode,
                creation_strategy=self.model_endpoint_creation_strategy,
                endpoint_type=self.endpoint_type,
            )

    def respond(self):
        """mark this step as the responder.

        step output will be returned as the flow result, no other step can follow
        """
        self.responder = True
        return self

    def run(self, event, *args, **kwargs):
        """run this step, in async flows the run is done through storey"""
        if not self._is_local_function(self.context):
            # todo invoke remote via REST call
            return event

        if self.context.verbose:
            self.context.logger.info(f"step {self.name} got event {event.body}")

        # inject context parameter if it is expected by the handler
        if self._inject_context:
            kwargs["context"] = self.context
        elif kwargs and "context" in kwargs:
            del kwargs["context"]

        try:
            if self.full_event or self._call_with_event:
                return self._handler(event, *args, **kwargs)

            if self._handler is None:
                raise MLRunInvalidArgumentError(
                    f"step {self.name} does not have a handler"
                )

            result = self._handler(
                _extract_input_data(self.input_path, event.body), *args, **kwargs
            )
            event.body = _update_result_body(self.result_path, event.body, result)
        except Exception as exc:
            if self._on_error_handler:
                self._log_error(event, exc)
                result = self._call_error_handler(event, exc)
                event.body = _update_result_body(self.result_path, event.body, result)
            else:
                raise exc
        return event

    def to_dict(
        self,
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
        strip: bool = False,
    ) -> dict:
        self.endpoint_type = (
            self.endpoint_type.value
            if isinstance(self.endpoint_type, schemas.EndpointType)
            else self.endpoint_type
        )
        self.model_endpoint_creation_strategy = (
            self.model_endpoint_creation_strategy.value
            if isinstance(
                self.model_endpoint_creation_strategy,
                schemas.ModelEndpointCreationStrategy,
            )
            else self.model_endpoint_creation_strategy
        )
        return super().to_dict(fields, exclude, strip)


class MonitoringApplicationStep(TaskStep):
    """monitoring application execution step, runs users class code"""

    kind = "monitoring_application"
    _default_class = ""

    def __init__(
        self,
        class_name: Optional[Union[str, type]] = None,
        class_args: Optional[dict] = None,
        handler: Optional[str] = None,
        name: Optional[str] = None,
        after: Optional[list] = None,
        full_event: Optional[bool] = None,
        function: Optional[str] = None,
        responder: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
    ):
        super().__init__(
            class_name=class_name,
            class_args=class_args,
            handler=handler,
            name=name,
            after=after,
            full_event=full_event,
            function=function,
            responder=responder,
            input_path=input_path,
            result_path=result_path,
        )


class ErrorStep(TaskStep):
    """error execution step, runs a class or handler"""

    kind = "error_step"
    _dict_fields = _task_step_fields + ["before", "base_step"]
    _default_class = ""

    def __init__(
        self,
        class_name: Optional[Union[str, type]] = None,
        class_args: Optional[dict] = None,
        handler: Optional[str] = None,
        name: Optional[str] = None,
        after: Optional[list] = None,
        full_event: Optional[bool] = None,
        function: Optional[str] = None,
        responder: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
    ):
        super().__init__(
            class_name=class_name,
            class_args=class_args,
            handler=handler,
            name=name,
            after=after,
            full_event=full_event,
            function=function,
            responder=responder,
            input_path=input_path,
            result_path=result_path,
        )
        self.before = None
        self.base_step = None


class RouterStep(TaskStep):
    """router step, implement routing logic for running child routes"""

    kind = "router"
    default_shape = "doubleoctagon"
    _dict_fields = _task_step_fields + ["routes", "name"]
    _default_class = "mlrun.serving.ModelRouter"

    def __init__(
        self,
        class_name: Optional[Union[str, type]] = None,
        class_args: Optional[dict] = None,
        handler: Optional[str] = None,
        routes: Optional[list] = None,
        name: Optional[str] = None,
        function: Optional[str] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
    ):
        super().__init__(
            class_name,
            class_args,
            handler,
            name=get_name(name, class_name or RouterStep.kind),
            function=function,
            input_path=input_path,
            result_path=result_path,
        )
        self._routes: ObjectDict = None
        self.routes = routes
        self.endpoint_type = schemas.EndpointType.ROUTER
        if isinstance(class_name, type):
            class_name = class_name.__name__
        self.model_endpoint_creation_strategy = (
            schemas.ModelEndpointCreationStrategy.INPLACE
            if class_name and "VotingEnsemble" in class_name
            else schemas.ModelEndpointCreationStrategy.SKIP
        )

    def get_children(self):
        """get child steps (routes)"""
        return self._routes.values()

    @property
    def routes(self):
        """child routes/steps, traffic is routed to routes based on router logic"""
        return self._routes

    @routes.setter
    def routes(self, routes: dict):
        self._routes = ObjectDict.from_dict(classes_map, routes, "task")

    def add_route(
        self,
        key,
        route=None,
        class_name=None,
        handler=None,
        function=None,
        creation_strategy: schemas.ModelEndpointCreationStrategy = schemas.ModelEndpointCreationStrategy.INPLACE,
        **class_args,
    ):
        """add child route step or class to the router, if key exists it will be updated

        :param key:        unique name (and route path) for the child step
        :param route:      child step object (Task, ..)
        :param class_name: class name to build the route step from (when route is not provided)
        :param class_args: class init arguments
        :param handler:    class handler to invoke on run/event
        :param function:   function this step should run in
        :param creation_strategy: Strategy for creating or updating the model endpoint:

                           * **overwrite**:

                           1. If model endpoints with the same name exist, delete the `latest` one.
                           2. Create a new model endpoint entry and set it as `latest`.

                           * **inplace** (default):

                           1. If model endpoints with the same name exist, update the `latest` entry.
                           2. Otherwise, create a new entry.

                           * **archive**:

                           1. If model endpoints with the same name exist, preserve them.
                           2. Create a new model endpoint with the same name and set it to `latest`.

        """

        if len(self.routes.keys()) >= MAX_MODELS_PER_ROUTER and key not in self.routes:
            raise mlrun.errors.MLRunModelLimitExceededError(
                f"Router cannot support more than {MAX_MODELS_PER_ROUTER} model endpoints. "
                f"To add a new route, edit an existing one by passing the same key."
            )
        if key in self.routes:
            logger.info(f"Model {key} already exists, updating it.")
        if not route and not class_name and not handler:
            raise MLRunInvalidArgumentError("route or class_name must be specified")
        if not route:
            route = TaskStep(
                class_name,
                class_args,
                handler=handler,
                model_endpoint_creation_strategy=creation_strategy,
                endpoint_type=schemas.EndpointType.LEAF_EP
                if self.class_name and "serving.VotingEnsemble" in self.class_name
                else schemas.EndpointType.NODE_EP,
            )
        route.function = function or route.function

        route = self._routes.update(key, route)
        route.set_parent(self)
        return route

    def clear_children(self, routes: list):
        """clear child steps (routes)"""
        if not routes:
            routes = self._routes.keys()
        for key in routes:
            del self._routes[key]

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        if not self.routes:
            raise mlrun.errors.MLRunRuntimeError(
                "You have to add models to the router step before initializing it"
            )
        if not self._is_local_function(context):
            return

        self.class_args = self.class_args or {}
        super().init_object(
            context, namespace, "skip", reset=reset, routes=self._routes, **extra_kwargs
        )

        for route in self._routes.values():
            if self.function and not route.function:
                # if the router runs on a child function and the
                # model function is not specified use the router function
                route.function = self.function
            route.set_parent(self)
            route.init_object(context, namespace, mode, reset=reset)

        self._set_error_handler()
        self._post_init(mode)

    def __getitem__(self, name):
        return self._routes[name]

    def __setitem__(self, name, route):
        self.add_route(name, route)

    def __delitem__(self, key):
        del self._routes[key]

    def __iter__(self):
        yield from self._routes.keys()

    def plot(self, filename=None, format=None, source=None, **kw):
        """plot/save graph using graphviz

        :param filename:  target filepath for the image (None for the notebook)
        :param format:    The output format used for rendering (``'pdf'``, ``'png'``, etc.)
        :param source:    source step to add to the graph
        :param kw:        kwargs passed to graphviz, e.g. rankdir="LR" (see: https://graphviz.org/doc/info/attrs.html)
        :return: graphviz graph object
        """
        return _generate_graphviz(
            self, _add_graphviz_router, filename, format, source=source, **kw
        )


class Model(storey.ParallelExecutionRunnable):
    def load(self) -> None:
        """Override to load model if needed."""
        pass

    def init(self):
        self.load()

    def predict(self, body: Any) -> Any:
        """Override to implement prediction logic. If the logic requires asyncio, override predict_async() instead."""
        return body

    async def predict_async(self, body: Any) -> Any:
        """Override to implement prediction logic if the logic requires asyncio."""
        return body

    def run(self, body: Any, path: str) -> Any:
        return self.predict(body)

    async def run_async(self, body: Any, path: str) -> Any:
        return self.predict(body)


class ModelSelector:
    """Used to select which models to run on each event."""

    def select(
        self, event, available_models: list[Model]
    ) -> Union[list[str], list[Model]]:
        """
        Given an event, returns a list of model names or a list of model objects to run on the event.
        If None is returned, all models will be run.

        :param event: The full event
        :param available_models: List of available models
        """
        pass


class ModelRunner(storey.ParallelExecution):
    """
    Runs multiple Models on each event. See ModelRunnerStep.

    :param model_selector: ModelSelector instance whose select() method will be used to select models to run on each
      event. Optional. If not passed, all models will be run.
    """

    def __init__(self, *args, model_selector: Optional[ModelSelector] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_selector = model_selector or ModelSelector()

    def select_runnables(self, event):
        models = cast(list[Model], self.runnables)
        return self.model_selector.select(event, models)


class ModelRunnerStep(TaskStep, StepToDict):
    """
    Runs multiple Models on each event.

    example::

        model_runner_step = ModelRunnerStep(name="my_model_runner")
        model_runner_step.add_model(MyModel(name="my_model"))
        graph.to(model_runner_step)

    :param model_selector: ModelSelector instance whose select() method will be used to select models to run on each
      event. Optional. If not passed, all models will be run.
    """

    kind = "model_runner"

    def __init__(
        self,
        *args,
        model_selector: Optional[Union[str, ModelSelector]] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            class_name="mlrun.serving.ModelRunner",
            class_args=dict(model_selector=model_selector),
            **kwargs,
        )

    def add_model(self, model: Union[str, Model], **model_parameters) -> None:
        """
        Add a Model to this ModelRunner.

        :param model: Model class name or object
        :param model_parameters: Parameters for model instantiation
        """
        models = self.class_args.get("models", [])
        models.append((model, model_parameters))
        self.class_args["models"] = models

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        model_selector = self.class_args.get("model_selector")
        models = self.class_args.get("models")
        if isinstance(model_selector, str):
            model_selector = get_class(model_selector, namespace)()
        model_objects = []
        for model, model_params in models:
            if not isinstance(model, Model):
                model = get_class(model, namespace)(**model_params)
            model_objects.append(model)
        self._async_object = ModelRunner(
            model_selector=model_selector,
            runnables=model_objects,
        )


class QueueStep(BaseStep, StepToDict):
    """queue step, implement an async queue or represent a stream"""

    kind = "queue"
    default_shape = "cds"
    _dict_fields = BaseStep._dict_fields + [
        "path",
        "shards",
        "retention_in_hours",
        "trigger_args",
        "options",
    ]

    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        after: Optional[list] = None,
        shards: Optional[int] = None,
        retention_in_hours: Optional[int] = None,
        trigger_args: Optional[dict] = None,
        **options,
    ):
        super().__init__(name, after)
        self.path = path
        self.shards = shards
        self.retention_in_hours = retention_in_hours
        self.options = options
        self.trigger_args = trigger_args
        self._stream = None
        self._async_object = None

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
        if self.path:
            self._stream = get_stream_pusher(
                self.path,
                shards=self.shards,
                retention_in_hours=self.retention_in_hours,
                **self.options,
            )
            if hasattr(self._stream, "create_stream"):
                self._stream.create_stream()
        self._set_error_handler()

    @property
    def async_object(self):
        return self._async_object

    def to(
        self,
        class_name: Union[str, StepToDict] = None,
        name: Optional[str] = None,
        handler: Optional[str] = None,
        graph_shape: Optional[str] = None,
        function: Optional[str] = None,
        full_event: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_endpoint_creation_strategy: Optional[
            schemas.ModelEndpointCreationStrategy
        ] = None,
        **class_args,
    ):
        if not function:
            name = get_name(name, class_name)
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"step '{name}' must specify a function, because it follows a queue step"
            )
        return super().to(
            class_name,
            name,
            handler,
            graph_shape,
            function,
            full_event,
            input_path,
            result_path,
            model_endpoint_creation_strategy,
            **class_args,
        )

    def run(self, event, *args, **kwargs):
        data = event.body
        if not data:
            return event

        if self._stream:
            full_event = self.options.get("full_event")
            if full_event or full_event is None and self.next:
                data = storey.utils.wrap_event_for_serialization(event, data)
            self._stream.push(data)
            event.terminated = True
            event.body = None
        return event


class FlowStep(BaseStep):
    """flow step, represent a workflow or DAG"""

    kind = "flow"
    _dict_fields = BaseStep._dict_fields + [
        "steps",
        "engine",
        "default_final_step",
    ]

    def __init__(
        self,
        name=None,
        steps=None,
        after: Optional[list] = None,
        engine=None,
        final_step=None,
    ):
        super().__init__(name, after)
        self._steps = None
        self.steps = steps
        self.engine = engine
        self.from_step = os.environ.get("START_FROM_STEP", None)
        self.final_step = final_step

        self._last_added = None
        self._controller = None
        self._wait_for_result = False
        self._source = None
        self._start_steps = []

    def get_children(self):
        return self._steps.values()

    @property
    def steps(self):
        """child (workflow) steps"""
        return self._steps

    @property
    def controller(self):
        """async (storey) flow controller"""
        return self._controller

    @steps.setter
    def steps(self, steps):
        self._steps = ObjectDict.from_dict(classes_map, steps, "task")

    def add_step(
        self,
        class_name=None,
        name=None,
        handler=None,
        after=None,
        before=None,
        graph_shape=None,
        function=None,
        full_event: Optional[bool] = None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_endpoint_creation_strategy: Optional[
            schemas.ModelEndpointCreationStrategy
        ] = None,
        **class_args,
    ):
        """add task, queue or router step/class to the flow

        use after/before to insert into a specific location

        example:
            graph = fn.set_topology("flow", exist_ok=True)
            graph.add_step(class_name="Chain", name="s1")
            graph.add_step(class_name="Chain", name="s3", after="$prev")
            graph.add_step(class_name="Chain", name="s2", after="s1", before="s3")

        :param class_name:  class name or step object to build the step from
                            for router steps the class name should start with '*'
                            for queue/stream step the class should be '>>' or '$queue'
        :param name:        unique name (and path) for the child step, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param after:       the step name this step comes after
                            can use $prev to indicate the last added step
        :param before:      string or list of next step names that will run after this step
        :param graph_shape: graphviz shape name
        :param function:    function this step should run in
        :param full_event:  this step accepts the full event (not just body)
        :param input_path:  selects the key/path in the event to use as input to the step
                            this require that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means the step will
                            receive 7 as input
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="y" means the output of the step will be written
                            to event["y"] resulting in {"x": 5, "y": <result>}
        :param model_endpoint_creation_strategy: Strategy for creating or updating the model endpoint:

                            * **overwrite**:

                            1. If model endpoints with the same name exist, delete the `latest` one.
                            2. Create a new model endpoint entry and set it as `latest`.

                            * **inplace** (default):

                            1. If model endpoints with the same name exist, update the `latest` entry.
                            2. Otherwise, create a new entry.

                            * **archive**:

                            1. If model endpoints with the same name exist, preserve them.
                            2. Create a new model endpoint with the same name and set it to `latest`.

        :param class_args:  class init arguments
        """

        name, step = params_to_step(
            class_name,
            name,
            handler,
            graph_shape=graph_shape,
            function=function,
            full_event=full_event,
            input_path=input_path,
            result_path=result_path,
            model_endpoint_creation_strategy=model_endpoint_creation_strategy,
            class_args=class_args,
        )

        after_list = after if isinstance(after, list) else [after]
        for after in after_list:
            self.insert_step(name, step, after, before)
        return step

    def insert_step(self, key, step, after, before=None):
        """insert step object into the flow, specify before and after"""

        step = self._steps.update(key, step)
        step.set_parent(self)

        if after == "$prev" and len(self._steps) == 1:
            after = None

        previous = ""
        if after:
            if after == "$prev" and self._last_added:
                previous = self._last_added.name
            else:
                if after not in self._steps.keys():
                    raise MLRunInvalidArgumentError(
                        f"cant set after, there is no step named {after}"
                    )
                previous = after
            step.after_step(previous)

        if before:
            if before not in self._steps.keys():
                raise MLRunInvalidArgumentError(
                    f"cant set before, there is no step named {before}"
                )
            if before == step.name or before == previous:
                raise GraphError(
                    f"graph loop, step {before} is specified in before and/or after {key}"
                )
            self[step.name].after_step(*self[before].after, append=False)
            self[before].after_step(step.name, append=False)
        self._last_added = step
        return step

    def clear_children(self, steps: Optional[list] = None):
        """remove some or all of the states, empty/None for all"""
        if not steps:
            steps = self._steps.keys()
        for key in steps:
            del self._steps[key]

    def __getitem__(self, name):
        return self._steps[name]

    def __setitem__(self, name, step):
        self.add_step(name, step)

    def __delitem__(self, key):
        del self._steps[key]

    def __iter__(self):
        yield from self._steps.keys()

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        """initialize graph objects and classes"""
        self.context = context
        self._insert_all_error_handlers()
        self.check_and_process_graph()

        for step in self._steps.values():
            step.set_parent(self)
            step.init_object(context, namespace, mode, reset=reset)
        self._set_error_handler()
        self._post_init(mode)

        if self.engine != "sync":
            self._build_async_flow()
            self._run_async_flow()

    def check_and_process_graph(self, allow_empty=False):
        """validate correct graph layout and initialize the .next links"""

        if self.is_empty() and allow_empty:
            self._start_steps = []
            return [], None, []

        def has_loop(step, previous):
            for next_step in step.after or []:
                if next_step in previous:
                    return step.name
                downstream = has_loop(self[next_step], previous + [next_step])
                if downstream:
                    return downstream
            return None

        start_steps = []
        for step in self._steps.values():
            step._next = None
            step._visited = False
            if step.after:
                loop_step = has_loop(step, [])
                if loop_step:
                    raise GraphError(
                        f"Error, loop detected in step {loop_step}, graph must be acyclic (DAG)"
                    )
            else:
                start_steps.append(step.name)

        responders = []
        for step in self._steps.values():
            if (
                hasattr(step, "responder")
                and step.responder
                and step.kind != "error_step"
            ):
                responders.append(step.name)
            if step.on_error and step.on_error in start_steps:
                start_steps.remove(step.on_error)
            if step.after:
                for prev_step in step.after:
                    self[prev_step].set_next(step.name)
        if self.on_error and self.on_error in start_steps:
            start_steps.remove(self.on_error)

        if (
            len(responders) > 1
        ):  # should not have multiple steps which respond to request
            raise GraphError(
                f'there are more than one responder steps in the graph ({",".join(responders)})'
            )

        if self.from_step:
            if self.from_step not in self.steps:
                raise GraphError(
                    f"from_step ({self.from_step}) specified and not found in graph steps"
                )
            start_steps = [self.from_step]

        self._start_steps = [self[name] for name in start_steps]

        def get_first_function_step(step, current_function):
            # find the first step which belongs to the function
            if (
                hasattr(step, "function")
                and step.function
                and step.function == current_function
            ):
                return step
            for item in step.next or []:
                next_step = self[item]
                returned_step = get_first_function_step(next_step, current_function)
                if returned_step:
                    return returned_step

        current_function = get_current_function(self.context)
        if current_function and current_function != "*":
            new_start_steps = []
            for from_step in self._start_steps:
                step = get_first_function_step(from_step, current_function)
                if step:
                    new_start_steps.append(step)
            if not new_start_steps:
                raise GraphError(
                    f"did not find steps pointing to current function ({current_function})"
                )
            self._start_steps = new_start_steps

        if self.engine == "sync" and len(self._start_steps) > 1:
            raise GraphError(
                "sync engine can only have one starting step (without .after)"
            )

        default_final_step = None
        if self.final_step:
            if self.final_step not in self.steps:
                raise GraphError(
                    f"final_step ({self.final_step}) specified and not found in graph steps"
                )
            default_final_step = self.final_step

        elif len(self._start_steps) == 1:
            # find the final step in case if a simple sequence of steps
            next_obj = self._start_steps[0]
            while next_obj:
                next = next_obj.next
                if not next:
                    default_final_step = next_obj.name
                    break
                next_obj = self[next[0]] if len(next) == 1 else None

        return self._start_steps, default_final_step, responders

    def set_flow_source(self, source):
        """set the async flow (storey) source"""
        self._source = source

    def _build_async_flow(self):
        """initialize and build the async/storey DAG"""

        def process_step(state, step, root):
            if not state._is_local_function(self.context) or state._visited:
                return
            for item in state.next or []:
                next_state = root[item]
                if next_state.async_object:
                    next_step = step.to(next_state.async_object)
                    process_step(next_state, next_step, root)
            state._visited = (
                True  # mark visited to avoid re-visit in case of multiple uplinks
            )

        default_source, self._wait_for_result = _init_async_objects(
            self.context, self._steps.values()
        )

        source = self._source or default_source
        for next_state in self._start_steps:
            next_step = source.to(next_state.async_object)
            process_step(next_state, next_step, self)

        for step in self._steps.values():
            # add error handler hooks
            if (step.on_error or self.on_error) and step.async_object:
                error_step = self._steps[step.on_error or self.on_error]
                # never set a step as its own error handler
                if step != error_step:
                    step.async_object.set_recovery_step(error_step.async_object)
                    for next_step in error_step.next or []:
                        next_state = self[next_step]
                        if next_state.async_object and error_step.async_object:
                            error_step.async_object.to(next_state.async_object)

        self._async_flow = source

    def _run_async_flow(self):
        self._controller = self._async_flow.run()

    def get_queue_links(self):
        """return dict of function and queue its listening on, for building stream triggers"""
        links = {}
        for step in self.get_children():
            if step.kind == StepKinds.queue:
                for item in step.next or []:
                    next_step = self[item]
                    if not next_step.function:
                        raise GraphError(
                            f"child function name must be specified in steps ({next_step.name}) which follow a queue"
                        )

                    if next_step.function in links:
                        raise GraphError(
                            f"function ({next_step.function}) cannot read from multiple queues"
                        )
                    links[next_step.function] = step
        return links

    def create_queue_streams(self):
        """create the streams used in this flow"""
        for step in self.get_children():
            if step.kind == StepKinds.queue:
                step.init_object(self.context, None)

    def list_child_functions(self):
        """return a list of child function names referred to in the steps"""
        functions = []
        for step in self.get_children():
            if (
                hasattr(step, "function")
                and step.function
                and step.function not in functions
            ):
                functions.append(step.function)
        return functions

    def is_empty(self):
        """is the graph empty (no child steps)"""
        return len(self.steps) == 0

    @staticmethod
    async def _await_and_return_id(awaitable, event):
        await awaitable
        event = copy(event)
        event.body = {"id": event.id}
        return event

    def run(self, event, *args, **kwargs):
        if self._controller:
            # async flow (using storey)
            event._awaitable_result = None
            if self.context.is_mock:
                resp = self._controller.emit(
                    event, return_awaitable_result=self._wait_for_result
                )
                if self._wait_for_result and resp:
                    return resp.await_result()
            else:
                resp_awaitable = self._controller.emit(
                    event, await_result=self._wait_for_result
                )
                if self._wait_for_result:
                    return resp_awaitable
                return self._await_and_return_id(resp_awaitable, event)
            event = copy(event)
            event.body = {"id": event.id}
            return event

        event = storey.utils.unpack_event_if_wrapped(event)

        if len(self._start_steps) == 0:
            return event
        next_obj = self._start_steps[0]
        while next_obj:
            try:
                event = next_obj.run(event, *args, **kwargs)
            except Exception as exc:
                if self._on_error_handler:
                    self._log_error(event, exc, failed_step=next_obj.name)
                    event.body = self._call_error_handler(event, exc)
                    event.terminated = True
                    return event
                else:
                    raise exc

            if hasattr(event, "terminated") and event.terminated:
                return event
            if (
                hasattr(event, "error")
                and isinstance(event.error, dict)
                and next_obj.name in event.error
            ):
                next_obj = self._steps[next_obj.on_error]
            next = next_obj.next
            if next and len(next) > 1:
                raise GraphError(
                    f"synchronous flow engine doesnt support branches use async, step={next_obj.name}"
                )
            next_obj = self[next[0]] if next else None
        return event

    def wait_for_completion(self):
        """wait for completion of run in async flows"""

        if self._controller:
            if hasattr(self._controller, "terminate"):
                return self._controller.terminate(wait=True)
            else:
                return self._controller.await_termination()

    def plot(self, filename=None, format=None, source=None, targets=None, **kw):
        """plot/save graph using graphviz

        :param filename:  target filepath for the graph image (None for the notebook)
        :param format:    the output format used for rendering (``'pdf'``, ``'png'``, etc.)
        :param source:    source step to add to the graph image
        :param targets:   list of target steps to add to the graph image
        :param kw:        kwargs passed to graphviz, e.g. rankdir="LR" (see https://graphviz.org/doc/info/attrs.html)
        :return:          graphviz graph object
        """
        return _generate_graphviz(
            self,
            _add_graphviz_flow,
            filename,
            format,
            source=source,
            targets=targets,
            **kw,
        )

    def _insert_all_error_handlers(self):
        """
        insert all error steps to the graph
        run after deployment
        """
        for name, step in self._steps.items():
            if step.kind == "error_step":
                self._insert_error_step(name, step)

    def _insert_error_step(self, name, step):
        """
        insert error step to the graph
        run after deployment
        """
        if not step.before and not any(
            [step.name in other_step.after for other_step in self._steps.values()]
        ):
            step.responder = True
            return

        for step_name in step.before:
            if step_name not in self._steps.keys():
                raise MLRunInvalidArgumentError(
                    f"cant set before, there is no step named {step_name}"
                )
            self[step_name].after_step(name)

    def set_flow(
        self,
        steps: list[Union[str, StepToDict, dict[str, Any]]],
        force: bool = False,
    ):
        if not force and self.steps:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "set_flow() called on a step that already has downstream steps. "
                "If you want to overwrite existing steps, set force=True."
            )

        self.steps = None
        step = self
        for next_step in steps:
            if isinstance(next_step, dict):
                step = step.to(**next_step)
            else:
                step = step.to(next_step)

        return step

    def supports_termination(self):
        return self.engine != "sync"


class RootFlowStep(FlowStep):
    """root flow step"""

    kind = "root"
    _dict_fields = ["steps", "engine", "final_step", "on_error"]


classes_map = {
    "task": TaskStep,
    "router": RouterStep,
    "flow": FlowStep,
    "queue": QueueStep,
    "error_step": ErrorStep,
    "monitoring_application": MonitoringApplicationStep,
    "model_runner": ModelRunnerStep,
}


def get_current_function(context):
    if context and hasattr(context, "current_function"):
        return context.current_function or ""
    return ""


def _add_graphviz_router(graph, step, source=None, **kwargs):
    if source:
        graph.node("_start", source.name, shape=source.shape, style="filled")
        graph.edge("_start", step.fullname)

    graph.node(step.fullname, label=step.name, shape=step.get_shape())
    for route in step.get_children():
        graph.node(route.fullname, label=route.name, shape=route.get_shape())
        graph.edge(step.fullname, route.fullname)


def _add_graphviz_flow(
    graph,
    step,
    source=None,
    targets=None,
):
    start_steps, default_final_step, responders = step.check_and_process_graph(
        allow_empty=True
    )
    graph.node("_start", source.name, shape=source.shape, style="filled")
    for start_step in start_steps:
        graph.edge("_start", start_step.fullname)
    for child in step.get_children():
        kind = child.kind
        if kind == StepKinds.router:
            with graph.subgraph(name="cluster_" + child.fullname) as sg:
                _add_graphviz_router(sg, child)
        else:
            graph.node(child.fullname, label=child.name, shape=child.get_shape())
        _add_edges(child.after or [], step, graph, child)
        _add_edges(getattr(child, "before", []), step, graph, child, after=False)
        if child.on_error:
            graph.edge(child.fullname, child.on_error, style="dashed")

    # draw targets after the last step (if specified)
    if targets:
        for target in targets or []:
            target_kind, target_name = target.name.split("/", 1)
            if target_kind != target_name:
                label = (
                    f"<{target_name}<br/><font point-size='8'>({target_kind})</font>>"
                )
            else:
                label = target_name
            graph.node(target.fullname, label=label, shape=target.get_shape())
            last_step = target.after or default_final_step
            if last_step:
                graph.edge(last_step, target.fullname)


def _add_edges(items, step, graph, child, after=True):
    for item in items:
        next_or_prev_object = step[item]
        kw = {}
        if next_or_prev_object.kind == StepKinds.router:
            kw["ltail"] = f"cluster_{next_or_prev_object.fullname}"
        if after:
            graph.edge(next_or_prev_object.fullname, child.fullname, **kw)
        else:
            graph.edge(child.fullname, next_or_prev_object.fullname, **kw)


def _generate_graphviz(
    step,
    renderer,
    filename=None,
    format=None,
    source=None,
    targets=None,
    **kw,
):
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            'graphviz is not installed, run "pip install graphviz" first!'
        )
    graph = Digraph("mlrun-flow", format="jpg")
    graph.attr(compound="true", **kw)
    source = source or BaseStep("start", shape="egg")
    renderer(graph, step, source=source, targets=targets)
    if filename:
        suffix = pathlib.Path(filename).suffix
        if suffix:
            filename = filename[: -len(suffix)]
            format = format or suffix[1:]
        format = format or "png"
        graph.render(filename, format=format)
    return graph


def graph_root_setter(server, graph):
    """set graph root object from class or dict"""
    if graph:
        if isinstance(graph, dict):
            kind = graph.get("kind")
        elif hasattr(graph, "kind"):
            kind = graph.kind
        else:
            raise MLRunInvalidArgumentError("graph must be a dict or a valid object")
        if kind == StepKinds.router:
            server._graph = server._verify_dict(graph, "graph", RouterStep)
        elif not kind or kind == StepKinds.root:
            server._graph = server._verify_dict(graph, "graph", RootFlowStep)
        else:
            raise GraphError(f"illegal root step {kind}")


def get_name(name, class_name):
    """get task name from provided name or class"""
    if name:
        return name
    if not class_name:
        raise MLRunInvalidArgumentError("name or class_name must be provided")
    if isinstance(class_name, type):
        return class_name.__name__
    return class_name.split(".")[-1]


def params_to_step(
    class_name,
    name,
    handler=None,
    graph_shape=None,
    function=None,
    full_event=None,
    input_path: Optional[str] = None,
    result_path: Optional[str] = None,
    class_args=None,
    model_endpoint_creation_strategy: Optional[
        schemas.ModelEndpointCreationStrategy
    ] = None,
    endpoint_type: Optional[schemas.EndpointType] = None,
):
    """return step object from provided params or classes/objects"""

    class_args = class_args or {}

    if isinstance(class_name, QueueStep):
        if not (name or class_name.name):
            raise MLRunInvalidArgumentError("queue name must be specified")

        step = class_name

    elif class_name in queue_class_names:
        if "path" not in class_args:
            raise MLRunInvalidArgumentError(
                "path=<stream path or None> must be specified for queues"
            )
        if not name:
            raise MLRunInvalidArgumentError("queue name must be specified")
        # Pass full_event on only if it's explicitly defined
        if full_event is not None:
            class_args = class_args.copy()
            class_args["full_event"] = full_event
        step = QueueStep(name, **class_args)

    elif class_name and hasattr(class_name, "to_dict"):
        struct = class_name.to_dict()
        kind = struct.get("kind", StepKinds.task)
        name = (
            name
            or struct.get("name", struct.get("class_name"))
            or class_name.to_dict(["name"]).get("name")
        )
        cls = classes_map.get(kind, RootFlowStep)
        step = cls.from_dict(struct)
        step.function = function
        step.full_event = full_event or step.full_event
        step.input_path = input_path or step.input_path
        step.result_path = result_path or step.result_path
        if kind == StepKinds.task:
            step.model_endpoint_creation_strategy = model_endpoint_creation_strategy
            step.endpoint_type = endpoint_type

    elif class_name and class_name.startswith("*"):
        routes = class_args.get("routes", None)
        class_name = class_name[1:]
        name = get_name(name, class_name or "router")
        step = RouterStep(
            class_name,
            class_args,
            handler,
            name=name,
            function=function,
            routes=routes,
            input_path=input_path,
            result_path=result_path,
        )

    elif class_name or handler:
        name = get_name(name, class_name)
        step = TaskStep(
            class_name,
            class_args,
            handler,
            name=name,
            function=function,
            full_event=full_event,
            input_path=input_path,
            result_path=result_path,
            model_endpoint_creation_strategy=model_endpoint_creation_strategy,
            endpoint_type=endpoint_type,
        )
    else:
        raise MLRunInvalidArgumentError("class_name or handler must be provided")

    if graph_shape:
        step.shape = graph_shape
    return name, step


def _init_async_objects(context, steps):
    try:
        import storey
    except ImportError:
        raise GraphError("storey package is not installed, use pip install storey")

    wait_for_result = False

    trigger = getattr(context, "trigger", None)
    context.logger.debug(f"trigger is {trigger or 'unknown'}")
    # respond is only supported for HTTP trigger
    respond_supported = trigger is None or trigger == "http"

    for step in steps:
        if hasattr(step, "async_object") and step._is_local_function(context):
            if step.kind == StepKinds.queue:
                skip_stream = context.is_mock and step.next
                if step.path and not skip_stream:
                    stream_path = step.path
                    endpoint = None
                    # in case of a queue, we default to a full_event=True
                    full_event = step.options.get("full_event")
                    options = {
                        "full_event": full_event or full_event is None and step.next
                    }
                    options.update(step.options)

                    kafka_brokers = get_kafka_brokers_from_dict(options, pop=True)

                    if stream_path and stream_path.startswith("ds://"):
                        datastore_profile = datastore_profile_read(stream_path)
                        if isinstance(
                            datastore_profile,
                            (DatastoreProfileKafkaTarget, DatastoreProfileKafkaSource),
                        ):
                            step._async_object = KafkaStoreyTarget(
                                path=stream_path,
                                context=context,
                                **options,
                            )
                        elif isinstance(datastore_profile, DatastoreProfileV3io):
                            step._async_object = StreamStoreyTarget(
                                stream_path=stream_path,
                                context=context,
                                **options,
                            )
                        else:
                            raise mlrun.errors.MLRunValueError(
                                f"Received an unexpected stream profile type: {type(datastore_profile)}\n"
                                "Expects `DatastoreProfileV3io` or `DatastoreProfileKafkaSource`."
                            )
                    elif stream_path.startswith("kafka://") or kafka_brokers:
                        topic, brokers = parse_kafka_url(stream_path, kafka_brokers)

                        kafka_producer_options = options.pop(
                            "kafka_producer_options", None
                        )

                        step._async_object = storey.KafkaTarget(
                            topic=topic,
                            brokers=brokers,
                            producer_options=kafka_producer_options,
                            context=context,
                            **options,
                        )
                    else:
                        if stream_path.startswith("v3io://"):
                            endpoint, stream_path = parse_path(step.path)
                            stream_path = stream_path.strip("/")
                        step._async_object = storey.StreamTarget(
                            storey.V3ioDriver(endpoint or config.v3io_api),
                            stream_path,
                            context=context,
                            **options,
                        )
                else:
                    step._async_object = storey.Map(lambda x: x)

            elif not step.async_object or not hasattr(step.async_object, "_outlets"):
                # if regular class, wrap with storey Map
                step._async_object = storey.Map(
                    step._handler,
                    full_event=step.full_event or step._call_with_event,
                    input_path=step.input_path,
                    result_path=step.result_path,
                    name=step.name,
                    context=context,
                    pass_context=step._inject_context,
                )
            if (
                respond_supported
                and not step.next
                and hasattr(step, "responder")
                and step.responder
            ):
                # if responder step (return result), add Complete()
                step.async_object.to(storey.Complete(full_event=True))
                wait_for_result = True

    source_args = context.get_param("source_args", {})
    explicit_ack = (
        is_explicit_ack_supported(context) and mlrun.mlconf.is_explicit_ack_enabled()
    )

    if context.is_mock:
        source_class = storey.SyncEmitSource
    else:
        source_class = storey.AsyncEmitSource

    default_source = source_class(
        context=context,
        explicit_ack=explicit_ack,
        **source_args,
    )
    return default_source, wait_for_result
