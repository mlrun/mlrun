# Copyright 2018 Iguazio
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

__all__ = ["TaskStep", "RouterStep", "RootFlowStep"]

import os
import pathlib
import traceback
import warnings
from copy import copy, deepcopy
from inspect import getfullargspec, signature
from typing import Union

from ..config import config
from ..datastore import get_stream_pusher
from ..errors import MLRunInvalidArgumentError
from ..model import ModelObj, ObjectDict
from ..platforms.iguazio import parse_v3io_path
from ..utils import get_class, get_function
from .utils import _extract_input_data, _update_result_body

callable_prefix = "_"
path_splitter = "/"
previous_step = "$prev"


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
]


def new_model_endpoint(class_name, model_path, handler=None, **class_args):
    class_args = deepcopy(class_args)
    class_args["model_path"] = model_path
    return TaskStep(class_name, class_args, handler=handler)


def new_remote_endpoint(url, **class_args):
    class_args = deepcopy(class_args)
    class_args["url"] = url
    return TaskStep("$remote", class_args)


class BaseStep(ModelObj):
    kind = "BaseStep"
    default_shape = "ellipse"
    _dict_fields = ["kind", "comment", "after", "on_error"]

    def __init__(self, name: str = None, after: list = None, shape: str = None):
        self.name = name
        self._parent = None
        self.comment = None
        self.context = None
        self.after = after
        self._next = None
        self.shape = shape
        self.on_error = None
        self._on_error_handler = None

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

    def after_step(self, after):
        """specify the previous step name"""
        # most steps only accept one source
        self.after = [after] if after else []
        return self

    def after_state(self, after):
        warnings.warn(
            "This method is deprecated. Use after_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.after_step(after)

    def error_handler(self, step_name: str = None, state_name=None):
        """set error handler step (on failure/raise of this step)"""
        if state_name:
            warnings.warn(
                "The state_name parameter is deprecated. Use step_name instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            step_name = step_name or state_name
        if not step_name:
            raise MLRunInvalidArgumentError("Must specify step_name")
        self.on_error = step_name
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
        self.context.logger.error(
            f"step {self.name} got error {err} when processing an event:\n {event.body}"
        )
        message = traceback.format_exc()
        self.context.logger.error(message)
        self.context.push_error(
            event, f"{err}\n{message}", source=self.fullname, **kwargs
        )

    def _call_error_handler(self, event, err, **kwargs):
        """call the error handler if exist"""
        if self._on_error_handler:
            event.error = str(err)
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

    def path_to_state(self, path: str):
        warnings.warn(
            "This method is deprecated. Use path_to_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.path_to_step(path)

    def to(
        self,
        class_name: Union[str, type] = None,
        name: str = None,
        handler: str = None,
        graph_shape: str = None,
        function: str = None,
        full_event: bool = None,
        input_path: str = None,
        result_path: str = None,
        **class_args,
    ):
        """add a step right after this step and return the new step

        example, a 4 step pipeline ending with a stream:
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
                            this require that the event body will behave like a dict, example:
                            event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means the step will
                            receive 7 as input
        :param result_path: selects the key/path in the event to write the results to
                            this require that the event body will behave like a dict, example:
                            event: {"x": 5} , result_path="y" means the output of the step will be written
                            to event["y"] resulting in {"x": 5, "y": <result>}
        :param class_args:  class init arguments
        """
        if hasattr(self, "steps"):
            parent = self
        elif self._parent:
            parent = self._parent
        else:
            raise GraphError(
                f"step {self.name} parent is not set or its not part of a graph"
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
        )
        step = parent._steps.update(name, step)
        step.set_parent(parent)
        if not hasattr(self, "steps"):
            # check that its not the root, todo: in future may gave nested flows
            step.after_step(self.name)
        parent._last_added = step
        return step


class TaskStep(BaseStep):
    """task execution step, runs a class or handler"""

    kind = "task"
    _dict_fields = _task_step_fields
    _default_class = ""

    def __init__(
        self,
        class_name: Union[str, type] = None,
        class_args: dict = None,
        handler: str = None,
        name: str = None,
        after: list = None,
        full_event: bool = None,
        function: str = None,
        responder: bool = None,
        input_path: str = None,
        result_path: str = None,
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
            return

        if isinstance(self.class_name, type):
            self._class_object = self.class_name
            self.class_name = self.class_name.__name__

        if not self._class_object:
            if self.class_name == "$remote":

                from mlrun.serving.remote import RemoteStep

                self._class_object = RemoteStep
            else:
                self._class_object = get_class(
                    self.class_name or self._default_class, namespace
                )

        if not self._object or reset:
            # init the step class + args
            class_args = {}
            for key, arg in self.class_args.items():
                if key.startswith(callable_prefix):
                    class_args[key[1:]] = get_function(arg, namespace)
                else:
                    class_args[key] = arg
            class_args.update(extra_kwargs)

            # add common args (name, context, ..) only if target class can accept them
            argspec = getfullargspec(self._class_object)
            for key in ["name", "context", "input_path", "result_path", "full_event"]:
                if argspec.varkw or key in argspec.args:
                    class_args[key] = getattr(self, key)

            try:
                self._object = self._class_object(**class_args)
            except TypeError as exc:
                raise TypeError(
                    f"failed to init step {self.name}, {exc}\n args={self.class_args}"
                )

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
            self._object.post_init(mode)
            if hasattr(self._object, "model_endpoint_uid"):
                self.endpoint_uid = self._object.model_endpoint_uid

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
            self._log_error(event, exc)
            handled = self._call_error_handler(event, exc)
            if not handled:
                raise exc
            event.terminated = True
        return event


class RouterStep(TaskStep):
    """router step, implement routing logic for running child routes"""

    kind = "router"
    default_shape = "doubleoctagon"
    _dict_fields = _task_step_fields + ["routes"]
    _default_class = "mlrun.serving.ModelRouter"

    def __init__(
        self,
        class_name: Union[str, type] = None,
        class_args: dict = None,
        handler: str = None,
        routes: list = None,
        name: str = None,
        function: str = None,
        input_path: str = None,
        result_path: str = None,
    ):
        super().__init__(
            class_name,
            class_args,
            handler,
            name=name,
            function=function,
            input_path=input_path,
            result_path=result_path,
        )
        self._routes: ObjectDict = None
        self.routes = routes

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
        **class_args,
    ):
        """add child route step or class to the router

        :param key:        unique name (and route path) for the child step
        :param route:      child step object (Task, ..)
        :param class_name: class name to build the route step from (when route is not provided)
        :param class_args: class init arguments
        :param handler:    class handler to invoke on run/event
        """

        if not route and not class_name:
            raise MLRunInvalidArgumentError("route or class_name must be specified")
        if not route:
            route = TaskStep(class_name, class_args, handler=handler)
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
        """plot/save a graphviz plot"""
        return _generate_graphviz(
            self, _add_graphviz_router, filename, format, source=source, **kw
        )


class QueueStep(BaseStep):
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
        name: str = None,
        path: str = None,
        after: list = None,
        shards: int = None,
        retention_in_hours: int = None,
        trigger_args: dict = None,
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
            )
        self._set_error_handler()

    @property
    def async_object(self):
        return self._async_object

    def after_step(self, after):
        # queue steps accept multiple sources
        if self.after:
            if after:
                self.after.append(after)
        else:
            self.after = [after] if after else []
        return self

    def after_state(self, after):
        warnings.warn(
            "This method is deprecated. Use after_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.after_step(after)

    def run(self, event, *args, **kwargs):
        data = event.body
        if not data:
            return event

        if self._stream:
            self._stream.push({"id": event.id, "body": data, "path": event.path})
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

    # TODO - remove once "states" is fully deprecated
    @classmethod
    def from_dict(cls, struct=None, fields=None, deprecated_fields: dict = None):
        deprecated_fields = deprecated_fields or {}
        deprecated_fields.update(
            {"states": "steps", "default_final_state": "default_final_step"}
        )

        return super().from_dict(
            struct, fields=fields, deprecated_fields=deprecated_fields
        )

    def __init__(
        self,
        name=None,
        steps=None,
        after: list = None,
        engine=None,
        final_step=None,
        # TODO - remove once usage of "state" is fully deprecated
        states=None,
        final_state=None,
    ):
        super().__init__(name, after)
        if states:
            warnings.warn(
                "The states parameter is deprecated. Use steps instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            steps = steps or states
        if final_state:
            warnings.warn(
                "The final_state parameter is deprecated. Use final_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            final_step = final_step or final_state

        self._steps = None
        self.steps = steps
        self.engine = engine
        # TODO - remove use of START_FROM_STATE once it's fully deprecated.
        self.from_step = os.environ.get("START_FROM_STEP", None) or os.environ.get(
            "START_FROM_STATE", None
        )
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
    def states(self):
        warnings.warn(
            "This property is deprecated. Use steps instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self._steps

    @property
    def controller(self):
        """async (storey) flow controller"""
        return self._controller

    @steps.setter
    def steps(self, steps):
        self._steps = ObjectDict.from_dict(classes_map, steps, "task")

    @states.setter
    def states(self, states):
        warnings.warn(
            "This property is deprecated. Use steps instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        self._steps = ObjectDict.from_dict(classes_map, states, "task")

    def add_step(
        self,
        class_name=None,
        name=None,
        handler=None,
        after=None,
        before=None,
        graph_shape=None,
        function=None,
        full_event: bool = None,
        input_path: str = None,
        result_path: str = None,
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
            class_args=class_args,
        )

        self.insert_step(name, step, after, before)
        return step

    def insert_state(self, key, state, after, before=None):
        warnings.warn(
            "This method is deprecated. Use insert_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        return self.insert_step(key, state, after, before)

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
            self[before].after_step(step.name)
        self._last_added = step
        return step

    def clear_children(self, steps: list = None, states: list = None):
        """remove some or all of the states, empty/None for all"""
        if states:
            warnings.warn(
                "This states parameter is deprecated. Use steps instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            steps = steps or states
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
        self.context = context
        self.check_and_process_graph()

        for step in self._steps.values():
            step.set_parent(self)
            step.init_object(context, namespace, mode, reset=reset)
        self._set_error_handler()
        self._post_init(mode)

        if self.engine != "sync":
            self._build_async_flow()

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
            if hasattr(step, "responder") and step.responder:
                responders.append(step.name)
            if step.on_error and step.on_error in start_steps:
                start_steps.remove(step.on_error)
            if step.after:
                prev_step = step.after[0]
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
            if not state._is_local_function(self.context):
                return
            for item in state.next or []:
                next_state = root[item]
                if next_state.async_object:
                    next_step = step.to(next_state.async_object)
                    process_step(next_state, next_step, root)

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

        self._controller = source.run()

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

    def init_queues(self):
        """init/create the streams used in this flow"""
        for step in self.get_children():
            if step.kind == StepKinds.queue:
                step.init_object(self.context, None)

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
            if config.datastore.async_source_mode == "enabled":
                resp_awaitable = self._controller.emit(
                    event, await_result=self._wait_for_result
                )
                if self._wait_for_result:
                    return resp_awaitable
                return self._await_and_return_id(resp_awaitable, event)
            else:
                resp = self._controller.emit(
                    event, return_awaitable_result=self._wait_for_result
                )
                if self._wait_for_result and resp:
                    return resp.await_result()
            event = copy(event)
            event.body = {"id": event.id}
            return event

        if len(self._start_steps) == 0:
            return event
        next_obj = self._start_steps[0]
        while next_obj:
            try:
                event = next_obj.run(event, *args, **kwargs)
            except Exception as exc:
                self._log_error(event, exc, failed_step=next_obj.name)
                handled = self._call_error_handler(event, exc)
                if not handled:
                    raise exc
                event.terminated = True
                return event

            if hasattr(event, "terminated") and event.terminated:
                return event
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
                self._controller.terminate()
            return self._controller.await_termination()

    def plot(self, filename=None, format=None, source=None, targets=None, **kw):
        """plot/save graph using graphviz"""
        return _generate_graphviz(
            self,
            _add_graphviz_flow,
            filename,
            format,
            source=source,
            targets=targets,
            **kw,
        )


class RootFlowStep(FlowStep):
    """root flow step"""

    kind = "root"
    _dict_fields = ["steps", "engine", "final_step", "on_error"]

    # TODO - remove once "final_state" is fully deprecated
    @classmethod
    def from_dict(cls, struct=None, fields=None):
        return super().from_dict(
            struct, fields=fields, deprecated_fields={"final_state": "final_step"}
        )


classes_map = {
    "task": TaskStep,
    "router": RouterStep,
    "flow": FlowStep,
    "queue": QueueStep,
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
    graph, step, source=None, targets=None,
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
        after = child.after or []
        for item in after:
            previous_object = step[item]
            kw = (
                {"ltail": "cluster_" + previous_object.fullname}
                if previous_object.kind == StepKinds.router
                else {}
            )
            graph.edge(previous_object.fullname, child.fullname, **kw)
        if child.on_error:
            graph.edge(child.fullname, child.on_error, style="dashed")

    # draw targets after the last step (if specified)
    if targets:
        for target in targets or []:
            graph.node(target.fullname, label=target.name, shape=target.get_shape())
            last_step = target.after or default_final_step
            if last_step:
                graph.edge(last_step, target.fullname)


def _generate_graphviz(
    step, renderer, filename=None, format=None, source=None, targets=None, **kw,
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
    return class_name


def params_to_state(
    class_name,
    name,
    handler=None,
    graph_shape=None,
    function=None,
    full_event=None,
    class_args=None,
):
    warnings.warn(
        "This method is deprecated. Use param_to_step instead",
        # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
        PendingDeprecationWarning,
    )
    return params_to_step(
        class_name, name, handler, graph_shape, function, full_event, class_args
    )


def params_to_step(
    class_name,
    name,
    handler=None,
    graph_shape=None,
    function=None,
    full_event=None,
    input_path: str = None,
    result_path: str = None,
    class_args=None,
):
    """return step object from provided params or classes/objects"""
    if class_name and hasattr(class_name, "to_dict"):
        struct = class_name.to_dict()
        kind = struct.get("kind", StepKinds.task)
        name = name or struct.get("name", struct.get("class_name"))
        cls = classes_map.get(kind, RootFlowStep)
        step = cls.from_dict(struct)
        step.function = function
        step.full_event = full_event or step.full_event
        step.input_path = input_path or step.input_path
        step.result_path = result_path or step.result_path

    elif class_name and class_name in [">>", "$queue"]:
        if "path" not in class_args:
            raise MLRunInvalidArgumentError(
                "path=<stream path or None> must be specified for queues"
            )
        if not name:
            raise MLRunInvalidArgumentError("queue name must be specified")
        step = QueueStep(name, **class_args)

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

    for step in steps:
        if hasattr(step, "async_object") and step._is_local_function(context):
            if step.kind == StepKinds.queue:
                skip_stream = context.is_mock and step.next
                if step.path and not skip_stream:
                    stream_path = step.path
                    endpoint = None
                    if "://" in stream_path:
                        endpoint, stream_path = parse_v3io_path(step.path)
                        stream_path = stream_path.strip("/")
                    step._async_object = storey.StreamTarget(
                        storey.V3ioDriver(endpoint), stream_path, context=context,
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
                )
            if not step.next and hasattr(step, "responder") and step.responder:
                # if responder step (return result), add Complete()
                step.async_object.to(storey.Complete(full_event=True))
                wait_for_result = True

    source_args = context.get_param("source_args", {})
    default_source = storey.SyncEmitSource(context=context, **source_args)
    return default_source, wait_for_result
