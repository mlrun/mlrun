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
from __future__ import annotations

import json
import os
import pathlib
import traceback
from copy import deepcopy, copy
from inspect import getmembers, isfunction, getfullargspec
from types import ModuleType
from typing import Union

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from ..platforms.iguazio import OutputStream
from ..model import ModelObj, ObjectDict
from ..utils import create_class, create_function

callable_prefix = "_"


class GraphError(Exception):
    """error in graph topology or configuration"""

    pass


class StateKinds:
    router = "router"
    task = "task"
    flow = "flow"
    queue = "queue"
    choice = "choice"
    root = "root"


_task_state_fields = [
    "kind",
    "class_name",
    "class_args",
    "handler",
    "skip_context",
    "next",
    "function",
    "comment",
    "shape",
    "full_event",
    "on_error",
]


def new_model_endpoint(class_name, model_path, handler=None, **class_args):
    class_args = deepcopy(class_args)
    class_args["model_path"] = model_path
    return TaskState(class_name, class_args, handler=handler)


def new_remote_endpoint(url, **class_args):
    class_args = deepcopy(class_args)
    class_args["url"] = url
    return TaskState("$remote", class_args)


class BaseState(ModelObj):
    kind = "BaseState"
    default_shape = "ellipse"
    _dict_fields = ["kind", "comment", "next", "on_error"]

    def __init__(self, name: str = None, next: str = None, shape: str = None):
        self.name = name
        self._parent = None
        self.comment = None
        self.context = None
        self.next = next
        self.shape = shape
        self.on_error = None
        self._on_error_handler = None

    def get_shape(self):
        """graphviz shape"""
        return self.shape or self.default_shape

    def set_parent(self, parent):
        """set/link the state parent (flow/router)"""
        self._parent = parent

    @property
    def parent(self):
        """state parent (flow/router)"""
        return self._parent

    def set_next(self, key: str, remove: str = None):
        """set/insert the key as next after this state, optionally remove other keys"""
        if not self.next:
            self.next = [key]
        elif key not in self.next:
            self.next.append(key)
        for state in remove or []:
            if state in self.next:
                self.next.remove(state)
        return self

    def error_handler(self, state_name: str):
        """set error handler state (on failure/raise of this state)"""
        self.on_error = state_name
        return self

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        """init the state class"""
        self.context = context

    def _is_local_function(self, context):
        return True

    def get_children(self):
        """get child states (for router/flow)"""
        return []

    def __iter__(self):
        yield from []

    @property
    def fullname(self):
        """full path/name (include parents)"""
        name = self.name or ""
        if self._parent and self._parent.fullname:
            name = ".".join([self._parent.fullname, name])
        return name.replace(":", "_")  # replace for graphviz escaping

    def _post_init(self, mode="sync"):
        pass

    def _set_error_handler(self):
        """init/link the error handler for this state"""
        if self.on_error:
            error_state = self.context.root.path_to_state(self.on_error)
            self._on_error_handler = error_state.run

    def _log_error(self, event, err, **kwargs):
        """on failure log"""
        self.context.logger.error(
            f"state {self.name} got error {err} when processing an event:\n {event.body}"
        )
        message = traceback.format_exc()
        self.context.logger.error(message)
        self.context.push_error(
            event, f"{err}\n{message}", source=self.fullname, **kwargs
        )

    def _call_error_handler(self, event, err, **kwargs):
        """call the error handler"""
        if self._on_error_handler:
            event.error = str(err)
            event.origin_state = self.fullname
            return self._on_error_handler(event)

    def path_to_state(self, path: str):
        """return state object from state relative/fullname"""
        path = path or ""
        tree = path.split(".")
        next_obj = self
        for state in tree:
            if state not in next_obj:
                raise GraphError(
                    f"step {state} doesnt exist in the graph under {next_obj.fullname}"
                )
            next_obj = next_obj[state]
        return next_obj

    def to(
        self,
        class_name: Union[str, TaskState, RouterState, QueueState, type] = None,
        name: str = None,
        handler: str = None,
        graph_shape: str = None,
        function: str = None,
        next_states: list = None,
        **class_args,
    ) -> Union[TaskState, QueueState, RouterState]:
        """add a state right after this state and return the new state

        example, a 4 step pipeline ending with a stream:
        graph.to('URLDownloader')\
             .to('ToParagraphs')\
             .to(name='to_json', handler='json.dumps')\
             .to('>', 'to_v3io', path=stream_path)\

        :param class_name:  class name or state object to build the state from
                            for router states the class name should start with '*'
                            for queue/stream state the class should be '>'
        :param name:        unique name (and path) for the child state, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param next_states: list of next step names that will run after this step
        :param graph_shape: graphviz shape name
        :param function:    function this state should run in
        :param class_args:  class init arguments
        """
        if hasattr(self, "start_at"):
            parent = self
        elif self._parent:
            parent = self._parent
        else:
            raise GraphError(
                f"state {self.name} parent is not set or its not part of a graph"
            )

        name, state = params_to_state(
            class_name,
            name,
            handler,
            graph_shape=graph_shape,
            function=function,
            class_args=class_args,
        )
        state = parent._states.update(name, state)
        state.set_parent(parent)
        if hasattr(self, "start_at"):
            self.start_at = state.name
        else:
            self.set_next(state.name, next_states)
        state.next = next_states
        parent._last_added = state
        return state


class TaskState(BaseState):
    kind = "task"
    _dict_fields = _task_state_fields
    _default_class = ""

    def __init__(
        self,
        class_name: Union[str, type] = None,
        class_args: dict = None,
        handler: str = None,
        name: str = None,
        next: list = None,
        full_event: bool = None,
        function: str = None,
    ):
        super().__init__(name, next)
        self.class_name = class_name
        self.class_args = class_args or {}
        self.handler = handler
        self.function = function
        self._handler = None
        self._object = None
        self.skip_context = None
        self.context = None
        self._class_object = None
        self.full_event = full_event
        self.on_error = None

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
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
            return

        if isinstance(self.class_name, type):
            self._class_object = self.class_name
            self.class_name = self.class_name.__name__

        if not self._class_object:
            if self.class_name == "$remote":
                self._class_object = RemoteHttpHandler
            else:
                self._class_object = get_class(
                    self.class_name or self._default_class, namespace
                )

        if not self._object or reset:
            # init the state class + args
            class_args = {}
            for key, arg in self.class_args.items():
                if key.startswith(callable_prefix):
                    class_args[key[1:]] = get_function(arg, namespace)
                else:
                    class_args[key] = arg
            class_args.update(extra_kwargs)

            # add name and context only if target class can accept them
            argspec = getfullargspec(self._class_object)
            if argspec.varkw or "context" in argspec.args:
                class_args["context"] = self.context
            if argspec.varkw or "name" in argspec.args:
                class_args["name"] = self.name

            try:
                self._object = self._class_object(**class_args)
            except TypeError as e:
                raise TypeError(
                    f"failed to init state {self.name}, {e}\n args={self.class_args}"
                )

            handler = self.handler
            if handler:
                if not hasattr(self._object, handler):
                    raise GraphError(
                        f"handler ({handler}) specified but doesnt exist in class {self.class_name}"
                    )
            else:
                if hasattr(self._object, "do"):
                    handler = "do"
                elif hasattr(self._object, "do_event"):
                    handler = "do_event"
                    self.full_event = True
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
    def object(self):
        return self._object

    def clear_object(self):
        self._object = None

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(mode)

    def respond(self):
        """mark this state as the responder.

        state output will be returned as the flow result, now other state can follow
        """
        if not self._parent or not hasattr(self._parent, "result_state"):
            raise GraphError(
                f"state {self.name} parent is not set or its not part of a flow"
            )
        self._parent.result_state = self.name
        self.next = None

    def run(self, event, *args, **kwargs):
        """run this state, in async flows the run is done through storey"""
        if not self._is_local_function(self.context):
            # todo invoke remote via REST call
            return event

        if self.context.verbose:
            self.context.logger.info(f"state {self.name} got event {event.body}")

        try:
            if self.full_event:
                return self._handler(event, *args, **kwargs)
            event.body = self._handler(event.body, *args, **kwargs)
        except Exception as e:
            self._log_error(event, e)
            handled = self._call_error_handler(event, e)
            if not handled:
                raise e
            event.terminated = True
        return event


class RouterState(TaskState):
    kind = "router"
    default_shape = "doubleoctagon"
    _dict_fields = _task_state_fields + ["routes"]
    _default_class = "mlrun.serving.ModelRouter"

    def __init__(
        self,
        class_name: Union[str, type] = None,
        class_args: dict = None,
        handler: str = None,
        routes: list = None,
        name: str = None,
        function: str = None,
    ):
        super().__init__(class_name, class_args, handler, name=name, function=function)
        self._routes: ObjectDict = None
        self.routes = routes

    def get_children(self):
        """get child states (routes)"""
        return self._routes.values()

    @property
    def routes(self):
        return self._routes

    @routes.setter
    def routes(self, routes: dict):
        self._routes = ObjectDict.from_dict(classes_map, routes, "task")

    def add_route(
        self, key, route=None, class_name=None, class_args=None, handler=None
    ):
        """add child route state or class to the router

        :param key:        unique name (and route path) for the child state
        :param route:      child state object (Task, ..)
        :param class_name: class name to build the route state from (when route is not provided)
        :param class_args: class init arguments
        :param handler:    class handler to invoke on run/event
        """

        if not route and not class_name:
            raise ValueError("route or class_name must be specified")
        if not route:
            route = TaskState(class_name, class_args, handler=handler)
        route = self._routes.update(key, route)
        route.set_parent(self)
        return route

    def add_model(
        self,
        name=None,
        model_path=None,
        class_name=None,
        model_url=None,
        handler=None,
        **class_args,
    ):
        """add ml model or route

        :param name:        model api key (or name:version), will determine the relative url/path
        :param model_path:  path to mlrun model artifact or model directory file/object path
        :param class_name:  V2 Model python class name
                            (can also module.submodule.class and it will be imported automatically)
        :param model_url:   url of a remote model serving endpoint (cannot be used with model_path)
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        if not model_path and not model_url:
            raise ValueError("model_path or model_url must be provided")
        class_name = class_name or self.spec.default_class
        if not isinstance(class_name, str):
            raise ValueError(
                "class name must be a string (name ot module.submodule.name)"
            )
        if model_path and not class_name:
            raise ValueError("model_path must be provided with class_name")
        if model_path:
            model_path = str(model_path)

        if model_url:
            state = new_remote_endpoint(model_url, **class_args)
        else:
            state = new_model_endpoint(class_name, model_path, handler, **class_args)

        self.add_route(name, state)
        return self

    def clear_children(self, routes: list):
        """clear child states (routes)"""
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
        source = source or BaseState("start", shape="egg")
        return _generate_graphviz(
            self, _add_gviz_router, filename, format, source=source, **kw
        )


class QueueState(BaseState):
    kind = "queue"
    default_shape = "cds"
    _dict_fields = BaseState._dict_fields + [
        "path",
        "shards",
        "retention_in_hours",
        "options",
    ]

    def __init__(
        self,
        name: str = None,
        path: str = None,
        next: list = None,
        shards: int = None,
        retention_in_hours: int = None,
        **options,
    ):
        super().__init__(name, next)
        self.path = path
        self.shards = shards
        self.retention_in_hours = retention_in_hours
        self.options = options
        self._stream = None

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
        if self.path:
            self._stream = OutputStream(self.path, self.shards, self.retention_in_hours)
        self._set_error_handler()

    def run(self, event, *args, **kwargs):
        data = event.body
        if not data:
            return event

        if self._stream:
            self._stream.push({"id": event.id, "body": data, "path": event.path})
            event.terminated = True
            event.body = None
        return event


class FlowState(BaseState):
    kind = "flow"
    _dict_fields = BaseState._dict_fields + [
        "states",
        "start_at",
        "engine",
        "result_state",
    ]

    def __init__(
        self,
        name=None,
        states=None,
        next=None,
        start_at=None,
        engine=None,
        result_state=None,
    ):
        super().__init__(name, next)
        self._states = None
        self.states = states
        self.start_at = start_at
        self.engine = engine
        self.from_state = os.environ.get("START_FROM_STATE", None)
        self.result_state = result_state

        self._last_added = None
        self._controller = None
        self._wait_for_result = False
        self._source = None

    def get_children(self):
        return self._states.values()

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        self._states = ObjectDict.from_dict(classes_map, states, "task")

    def add_step(
        self,
        class_name=None,
        name=None,
        handler=None,
        after=None,
        before=None,
        graph_shape=None,
        function=None,
        **class_args,
    ):
        """add task, queue or router state/class to the flow

        use after/before to insert into a specific location

        example:
            graph = fn.set_topology("flow", exist_ok=True)
            graph.add_step(class_name="Chain", name="s1", after="$start")
            graph.add_step(class_name="Chain", name="s3", after="$last")
            graph.add_step(class_name="Chain", name="s2", after="s1", before="s3")

        :param class_name:  class name or state object to build the state from
                            for router states the class name should start with '*'
                            for queue/stream state the class should be '>'
        :param name:        unique name (and path) for the child state, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param after:       the step name this step comes after
                            can use control strings: $start, $prev, $last
        :param before:      string or list of next step names that will run after this step
        :param graph_shape: graphviz shape name
        :param function:    function this state should run in
        :param class_args:  class init arguments
        """

        name, state = params_to_state(
            class_name,
            name,
            handler,
            graph_shape=graph_shape,
            function=function,
            class_args=class_args,
        )

        if not after and not before:
            after = '$prev'

        self.insert_state(name, state, after, before)
        return state

    def insert_state(self, key, state, after, before=None, shape=None):
        """insert state object into the flow, specify before and after"""

        if after == "$last" and before:
            raise ValueError("cannot specify after $last and before state(s) together")

        state = self._states.update(key, state)
        state.set_parent(self)

        if before:
            if not isinstance(before, list):
                before = [before]
            state.next = before

        if after == "$prev" and len(self._states) == 1:
            after = "$start"

        # re adjust start_at
        if (
            after == "$start"
            or (not self.start_at and after in ["$prev", "$last"])
            or (before and self.start_at in before)
        ):
            if (
                after == "$start"
                and not before
                and self.start_at
                and self.start_at != state.name
            ):
                # move the previous start_at to be after our step
                state.next = [self.start_at]
            self.start_at = state.name

        after_state = None
        if after and not after.startswith("$"):
            if after not in self._states.keys():
                raise ValueError(f"there is no state named {after}")
            after_state = self._states[after]
        if after == "$prev":
            after_state = self._last_added
        elif after == "$last":
            name = self.find_last_state()
            if name and name != state.name:
                after_state = self._states[name]

        if after_state:
            if before and after_state.name in before:
                raise GraphError(
                    f"graph loop, state {after_state.name} is specified in before and after {key}"
                )
            after_state.set_next(state.name, before)
        self._last_added = state
        return state

    def clear_children(self, states: list = None):
        """remove some or all of the states, empty/None for all"""
        if not states:
            states = self._states.keys()
        for key in states:
            del self._states[key]

    def __getitem__(self, name):
        return self._states[name]

    def __setitem__(self, name, state):
        self.add_step(name, state)

    def __delitem__(self, key):
        del self._states[key]

    def __iter__(self):
        yield from self._states.keys()

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
        loop_state = self.detect_loops()
        if loop_state:
            raise GraphError(
                f"Error, loop detected in state {loop_state}, graph must be acyclic (DAG)"
            )

        for state in self._states.values():
            state.set_parent(self)
            state.init_object(context, namespace, mode, reset=reset)
        self._set_error_handler()
        self._post_init(mode)

        if self.engine in ["storey", "async"]:
            self._build_async_flow()

    def set_flow_source(self, source):
        self._source = source

    def _build_async_flow(self):
        import storey

        def get_step(state):
            if state.kind == StateKinds.queue:
                if state.path:
                    return storey.WriteToV3IOStream(storey.V3ioDriver(), state.path)
                else:
                    return storey.Map(lambda x: x)
            is_storey = (
                hasattr(state, "object")
                and state.object
                and hasattr(state.object, "_outlets")
            )
            if is_storey:
                return state.object
            # todo Q state & choice
            return storey.Map(state.run, full_event=True)

        def process_step(state, step, root):
            if not state._is_local_function(self.context):
                return

            if not state.next and self.result_state == state.name:
                # print("set complete:", state.name)
                step.to(storey.Complete(full_event=True))
                self._wait_for_result = True
                return
            for item in state.next or []:
                # print("visit:", state.name, item)
                next_state = root[item]
                next_step = step.to(get_step(next_state))
                process_step(next_state, next_step, root)

        next_state = self.get_start_state()
        # todo: allow source array (e.g. data->json loads..)
        source = self._source or storey.Source()
        next_step = source.to(get_step(next_state))
        process_step(next_state, next_step, self)
        self._controller = source.run()

    def detect_loops(self):
        """find loops in the graph"""

        def has_loop(state, previous):
            for next_state in state.next or []:
                if next_state in previous:
                    return state.name
                downstream = has_loop(self[next_state], previous + [next_state])
                if downstream:
                    return downstream
            return None

        return has_loop(self.get_start_state(), [])

    def get_start_state(self, from_state=None):
        def get_first_function_state(state, current_function):
            if (
                hasattr(state, "function")
                and state.function
                and state.function == current_function
            ):
                return state
            for item in state.next or []:
                next_state = self[item]
                resp = get_first_function_state(next_state, current_function)
                if resp:
                    return resp

        from_state = from_state or self.from_state or self.start_at
        current_function = get_current_function(self.context)
        if current_function:
            state = get_first_function_state(self[from_state], current_function)
            if not state:
                raise ValueError(
                    f"states not found pointing to function {current_function}"
                )
            return state

        if not from_state:
            raise ValueError("start step was not specified in flow")
        return self.path_to_state(from_state)

    def get_queue_links(self):
        """return dict of function and queue its listening on, for building stream triggers"""
        links = {}
        for state in self.get_children():
            if state.kind == StateKinds.queue:
                for item in state.next or []:
                    next_state = self[item]
                    if next_state.function:
                        if next_state.function in links:
                            raise GraphError(
                                f"function ({next_state.function}) cannot read from multiple queues"
                            )
                        links[next_state.function] = state
        return links

    def init_queues(self):
        """init/create the streams used in this flow"""
        for state in self.get_children():
            if state.kind == StateKinds.queue:
                state.init_object(self.context, None)

    def find_last_state(self):
        if self.result_state:
            return self.result_state

        loop_state = self.detect_loops()
        if loop_state:
            raise GraphError(
                f"Error, loop detected in state {loop_state}, graph must be acyclic (DAG)"
            )

        next_obj = self.get_start_state()
        while next_obj:
            next = next_obj.next
            if not next:
                return next_obj.name
            next_obj = self[next[0]]

    def run(self, event, *args, **kwargs):

        if self._controller:
            # async flow (using storey)
            event._awaitable_result = None
            resp = self._controller.emit(
                event, return_awaitable_result=self._wait_for_result
            )
            if self._wait_for_result:
                return resp.await_result()
            event = copy(event)
            event.body = {"id": event.id}
            return event

        next_obj = self.get_start_state(kwargs.get("from_state", None))
        while next_obj:
            try:
                event = next_obj.run(event, *args, **kwargs)
            except Exception as e:
                self._log_error(event, e, failed_state=next_obj.name)
                handled = self._call_error_handler(event, e)
                if not handled:
                    raise e
                event.terminated = True
                return event

            if hasattr(event, "terminated") and event.terminated:
                return event
            next = next_obj.next
            if next and len(next) > 1:
                raise GraphError(
                    f"synchronous flow engine doesnt support branches use async, state={next_obj.name}"
                )
            next_obj = self[next[0]] if next else None
        return event

    def wait_for_completion(self):
        """wait for completion of run in async flows"""
        if self._controller:
            self._controller.terminate()
            self._controller.await_termination()

    def plot(self, filename=None, format=None, source=None, targets=None, **kw):
        """plot/save graph using graphviz"""
        return _generate_graphviz(
            self, _add_gviz_flow, filename, format, source=source, targets=targets, **kw
        )


class RootFlowState(FlowState):
    kind = "root"
    _dict_fields = ["states", "start_at", "engine", "result_state", "on_error"]


http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class RemoteHttpHandler:
    """class for calling remote endpoints"""

    def __init__(self, url):
        self.url = url
        self.format = "json"
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._session.mount("https://", http_adapter)

    def do_event(self, event):
        kwargs = {}
        kwargs["headers"] = event.headers or {}
        method = event.method or "POST"
        if method != "GET":
            if isinstance(event.body, (str, bytes)):
                kwargs["data"] = event.body
            else:
                kwargs["json"] = event.body

        url = self.url.strip("/") + event.path
        try:
            resp = self._session.request(method, url, verify=False, **kwargs)
        except OSError as err:
            raise OSError(f"error: cannot run function at url {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.text}")

        data = resp.content
        if self.format == "json" or resp.headers["content-type"] == "application/json":
            data = json.loads(data)
        event.body = data
        return event


def _module_to_namespace(namespace):
    if isinstance(namespace, ModuleType):
        members = getmembers(namespace, lambda o: isfunction(o) or isinstance(o, type))
        return {key: mod for key, mod in members}
    return namespace


def get_class(class_name, namespace):
    """return class object from class name string"""
    if isinstance(class_name, type):
        return class_name
    namespace = _module_to_namespace(namespace)
    if class_name in namespace:
        return namespace[class_name]

    try:
        class_object = create_class(class_name)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, class {class_name} not found, {e}")
    return class_object


def get_function(function, namespace):
    """return function callable object from function name string"""
    if callable(function):
        return function

    function = function.strip()
    if function.startswith("("):
        if not function.endswith(")"):
            raise ValueError('function expression must start with "(" and end with ")"')
        return eval("lambda event: " + function[1:-1], {}, {})
    namespace = _module_to_namespace(namespace)
    if function in namespace:
        return namespace[function]

    try:
        function_object = create_function(function)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, function {function} not found, {e}")
    return function_object


classes_map = {
    "task": TaskState,
    "router": RouterState,
    "flow": FlowState,
    "queue": QueueState,
}


def get_current_function(context):
    if context and hasattr(context, "current_function"):
        return context.current_function or ""
    return ""


def _add_gviz_router(g, state, source=None, **kwargs):
    if source:
        g.node("_start", source.name, shape=source.shape, style="filled")
        g.edge("_start", state.fullname)

    g.node(state.fullname, label=state.name, shape=state.get_shape())
    for route in state.get_children():
        g.node(route.fullname, label=route.name, shape=route.get_shape())
        g.edge(state.fullname, route.fullname)


def _add_gviz_flow(
    g, state, source=None, targets=None,
):
    source = source or BaseState("start", shape="egg")
    g.node("_start", source.name, shape=source.shape, style="filled")
    g.edge("_start", state.start_at)
    for child in state.get_children():
        kind = child.kind
        if kind == StateKinds.router:
            with g.subgraph(name="cluster_" + child.fullname) as sg:
                _add_gviz_router(sg, child)
        else:
            g.node(child.fullname, label=child.name, shape=child.get_shape())
        next = child.next or []
        for item in next:
            next_object = state[item]
            kw = (
                {"ltail": "cluster_" + child.fullname}
                if child.kind == StateKinds.router
                else {}
            )
            g.edge(child.fullname, next_object.fullname, **kw)

    # draw targets after the last state (if specified)
    if targets:
        last_state = state.find_last_state()
        for target in targets or []:
            g.node(target.fullname, label=target.name, shape=target.get_shape())
            g.edge(last_state, target.fullname)


def _generate_graphviz(
    state, renderer, filename=None, format=None, source=None, targets=None, **kw,
):
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            'graphviz is not installed, run "pip install graphviz" first!'
        )
    g = Digraph("mlrun-flow", format="jpg")
    g.attr(compound="true", **kw)
    renderer(g, state, source=source, targets=targets)
    if filename:
        suffix = pathlib.Path(filename).suffix
        if suffix:
            filename = filename[: -len(suffix)]
            format = format or suffix[1:]
        format = format or "png"
        g.render(filename, format=format)
    return g


def graph_root_setter(server, graph):
    if graph:
        if isinstance(graph, dict):
            kind = graph.get("kind")
        elif hasattr(graph, "kind"):
            kind = graph.kind
        else:
            raise ValueError("graph must be a dict or a valid object")
        if kind == StateKinds.router:
            server._graph = server._verify_dict(graph, "graph", RouterState)
        elif not kind or kind == StateKinds.root:
            server._graph = server._verify_dict(graph, "graph", RootFlowState)
        else:
            raise GraphError(f'illegal root state {kind}')


def get_name(name, class_name):
    if name:
        return name
    if not class_name:
        raise ValueError("name or class_name must be provided")
    if isinstance(class_name, type):
        return class_name.__name__
    return class_name


def params_to_state(
    class_name, name, handler=None, graph_shape=None, function=None, class_args=None
):
    if class_name and hasattr(class_name, "to_dict"):
        struct = class_name.to_dict()
        kind = struct.get('kind', StateKinds.task)
        name = name or struct.get('name', struct.get('class_name'))
        cls = classes_map.get(kind, RootFlowState)
        state = cls.from_dict(struct)
        state.function = function
    elif class_name and class_name == ">":
        if "path" not in class_args:
            raise ValueError("path=<stream path or None> must be specified for queues")
        state = QueueState(name, **class_args)
    elif class_name and class_name.startswith("*"):
        routes = class_args.get("routes", None)
        class_name = class_name[1:]
        name = get_name(name, class_name or "router")
        state = RouterState(
            class_name, class_args, handler, name=name, function=function, routes=routes
        )
    elif class_name or handler:
        name = get_name(name, class_name)
        state = TaskState(class_name, class_args, handler, name=name, function=function)
    else:
        raise ValueError("class_name or handler must be provided")

    if graph_shape:
        state.shape = graph_shape
    return name, state
