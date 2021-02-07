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

__all__ = ["TaskState", "RouterState", "RootFlowState"]

import json
import os
import pathlib
import traceback
from copy import deepcopy, copy
from inspect import getfullargspec
from typing import Union

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from ..datastore import get_stream_pusher
from ..model import ModelObj, ObjectDict
from ..utils import get_function, get_class
from ..errors import MLRunInvalidArgumentError

callable_prefix = "_"
path_splitter = "/"
previous_step = "$prev"


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
    "after",
    "function",
    "comment",
    "shape",
    "full_event",
    "on_error",
    "responder",
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
        """set/link the state parent (flow/router)"""
        self._parent = parent

    @property
    def next(self):
        return self._next

    @property
    def parent(self):
        """state parent (flow/router)"""
        return self._parent

    def set_next(self, key: str):
        """set/insert the key as next after this state, optionally remove other keys"""
        if not self.next:
            self._next = [key]
        elif key not in self.next:
            self._next.append(key)
        return self

    def after_state(self, after):
        """specify the previous state name"""
        # most states only accept one source
        self.after = [after] if after else []
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
            name = path_splitter.join([self._parent.fullname, name])
        return name.replace(":", "_")  # replace for graphviz escaping

    def _post_init(self, mode="sync"):
        pass

    def _set_error_handler(self):
        """init/link the error handler for this state"""
        if self.on_error:
            error_state = self.context.root.path_to_state(self.on_error)
            self._on_error_handler = error_state.run

    def _log_error(self, event, err, **kwargs):
        """on failure log (for sync mode)"""
        self.context.logger.error(
            f"state {self.name} got error {err} when processing an event:\n {event.body}"
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

    def path_to_state(self, path: str):
        """return state object from state relative/fullname"""
        path = path or ""
        tree = path.split(path_splitter)
        next_level = self
        for state in tree:
            if state not in next_level:
                raise GraphError(
                    f"step {state} doesnt exist in the graph under {next_level.fullname}"
                )
            next_level = next_level[state]
        return next_level

    def to(
        self,
        class_name: Union[str, type] = None,
        name: str = None,
        handler: str = None,
        graph_shape: str = None,
        function: str = None,
        full_event: bool = None,
        **class_args,
    ):
        """add a state right after this state and return the new state

        example, a 4 step pipeline ending with a stream:
        graph.to('URLDownloader')\
             .to('ToParagraphs')\
             .to(name='to_json', handler='json.dumps')\
             .to('>', 'to_v3io', path=stream_path)\

        :param class_name:  class name or state object to build the state from
                            for router states the class name should start with '*'
                            for queue/stream state the class should be '>>' or '$queue'
        :param name:        unique name (and path) for the child state, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param graph_shape: graphviz shape name
        :param function:    function this state should run in
        :param full_event:  this step accepts the full event (not just body)
        :param class_args:  class init arguments
        """
        if hasattr(self, "states"):
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
            full_event=full_event,
            class_args=class_args,
        )
        state = parent._states.update(name, state)
        state.set_parent(parent)
        if not hasattr(self, "states"):
            # check that its not the root, todo: in future may gave nested flows
            state.after_state(self.name)
        parent._last_added = state
        return state


class TaskState(BaseState):
    """task execution state, runs a class or handler"""

    kind = "task"
    _dict_fields = _task_state_fields
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
        self.on_error = None

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

            # determine the right class handler to use
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
    def async_object(self):
        """return the sync or async (storey) class instance"""
        return self._async_object or self._object

    def clear_object(self):
        self._object = None

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(mode)

    def respond(self):
        """mark this state as the responder.

        state output will be returned as the flow result, no other state can follow
        """
        self.responder = True
        return self

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
    """router state, implement routing logic for running child routes"""

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
        """child routes/states, traffic is routed to routes based on router logic"""
        return self._routes

    @routes.setter
    def routes(self, routes: dict):
        self._routes = ObjectDict.from_dict(classes_map, routes, "task")

    def add_route(self, key, route=None, class_name=None, handler=None, **class_args):
        """add child route state or class to the router

        :param key:        unique name (and route path) for the child state
        :param route:      child state object (Task, ..)
        :param class_name: class name to build the route state from (when route is not provided)
        :param class_args: class init arguments
        :param handler:    class handler to invoke on run/event
        """

        if not route and not class_name:
            raise MLRunInvalidArgumentError("route or class_name must be specified")
        if not route:
            route = TaskState(class_name, class_args, handler=handler)
        route = self._routes.update(key, route)
        route.set_parent(self)
        return route

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
        return _generate_graphviz(
            self, _add_graphviz_router, filename, format, source=source, **kw
        )


class QueueState(BaseState):
    """queue state, implement an async queue or represent a stream"""

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
        after: list = None,
        shards: int = None,
        retention_in_hours: int = None,
        **options,
    ):
        super().__init__(name, after)
        self.path = path
        self.shards = shards
        self.retention_in_hours = retention_in_hours
        self.options = options
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

    def after_state(self, after):
        # queue states accept multiple sources
        if self.after:
            if after:
                self.after.append(after)
        else:
            self.after = [after] if after else []
        return self

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
    """flow state, represent a workflow or DAG"""

    kind = "flow"
    _dict_fields = BaseState._dict_fields + [
        "states",
        "engine",
        "default_final_state",
    ]

    def __init__(
        self, name=None, states=None, after: list = None, engine=None, final_state=None,
    ):
        super().__init__(name, after)
        self._states = None
        self.states = states
        self.engine = engine
        self.from_state = os.environ.get("START_FROM_STATE", None)
        self.final_state = final_state

        self._last_added = None
        self._controller = None
        self._wait_for_result = False
        self._source = None
        self._start_states = []

    def get_children(self):
        return self._states.values()

    @property
    def states(self):
        """child (workflow) states"""
        return self._states

    @property
    def controller(self):
        """async (storey) flow controller"""
        return self._controller

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
        full_event: bool = None,
        **class_args,
    ):
        """add task, queue or router state/class to the flow

        use after/before to insert into a specific location

        example:
            graph = fn.set_topology("flow", exist_ok=True)
            graph.add_step(class_name="Chain", name="s1")
            graph.add_step(class_name="Chain", name="s3", after="$prev")
            graph.add_step(class_name="Chain", name="s2", after="s1", before="s3")

        :param class_name:  class name or state object to build the state from
                            for router states the class name should start with '*'
                            for queue/stream state the class should be '>>' or '$queue'
        :param name:        unique name (and path) for the child state, default is class name
        :param handler:     class/function handler to invoke on run/event
        :param after:       the step name this step comes after
                            can use $prev to indicate the last added state
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
            full_event=full_event,
            class_args=class_args,
        )

        self.insert_state(name, state, after, before)
        return state

    def insert_state(self, key, state, after, before=None):
        """insert state object into the flow, specify before and after"""

        state = self._states.update(key, state)
        state.set_parent(self)

        if after == "$prev" and len(self._states) == 1:
            after = None

        previous = ""
        if after:
            if after == "$prev" and self._last_added:
                previous = self._last_added.name
            else:
                if after not in self._states.keys():
                    raise MLRunInvalidArgumentError(
                        f"cant set after, there is no state named {after}"
                    )
                previous = after
            state.after_state(previous)

        if before:
            if before not in self._states.keys():
                raise MLRunInvalidArgumentError(
                    f"cant set before, there is no state named {before}"
                )
            if before == state.name or before == previous:
                raise GraphError(
                    f"graph loop, state {before} is specified in before and/or after {key}"
                )
            self[before].after_state(state.name)
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
        self.check_and_process_graph()

        for state in self._states.values():
            state.set_parent(self)
            state.init_object(context, namespace, mode, reset=reset)
        self._set_error_handler()
        self._post_init(mode)

        if self.engine != "sync":
            self._build_async_flow()

    def check_and_process_graph(self, allow_empty=False):
        """validate correct graph layout and initialize the .next links"""

        if self.is_empty() and allow_empty:
            self._start_states = []
            return [], None, []

        def has_loop(state, previous):
            for next_state in state.after or []:
                if next_state in previous:
                    return state.name
                downstream = has_loop(self[next_state], previous + [next_state])
                if downstream:
                    return downstream
            return None

        start_states = []
        for state in self._states.values():
            state._next = None
            if state.after:
                loop_state = has_loop(state, [])
                if loop_state:
                    raise GraphError(
                        f"Error, loop detected in state {loop_state}, graph must be acyclic (DAG)"
                    )
            else:
                start_states.append(state.name)

        responders = []
        for state in self._states.values():
            if hasattr(state, "responder") and state.responder:
                responders.append(state.name)
            if state.on_error and state.on_error in start_states:
                start_states.remove(state.on_error)
            if state.after:
                prev_state = state.after[0]
                self[prev_state].set_next(state.name)
        if self.on_error and self.on_error in start_states:
            start_states.remove(self.on_error)

        if (
            not start_states
        ):  # for safety, not sure if its possible to get here (since its a loop)
            raise GraphError("there are no starting states (ones without .after)")

        if (
            len(responders) > 1
        ):  # should not have multiple steps which respond to request
            raise GraphError(
                f'there are more than one responder states in the graph ({",".join(responders)})'
            )

        if self.from_state:
            if self.from_state not in self.states:
                raise GraphError(
                    f"from_state ({self.from_state}) specified and not found in graph states"
                )
            start_states = [self.from_state]

        self._start_states = [self[name] for name in start_states]

        def get_first_function_state(state, current_function):
            # find the first state which belongs to the function
            if (
                hasattr(state, "function")
                and state.function
                and state.function == current_function
            ):
                return state
            for item in state.next or []:
                next_state = self[item]
                returned_state = get_first_function_state(next_state, current_function)
                if returned_state:
                    return returned_state

        current_function = get_current_function(self.context)
        if current_function:
            new_start_states = []
            for from_state in self._start_states:
                state = get_first_function_state(from_state, current_function)
                if state:
                    new_start_states.append(state)
            if not new_start_states:
                raise GraphError(
                    f"did not find states pointing to current function ({current_function})"
                )
            self._start_states = new_start_states

        if self.engine == "sync" and len(self._start_states) > 1:
            raise GraphError(
                "sync engine can only have one starting state (without .after)"
            )

        default_final_state = None
        if self.final_state:
            if self.final_state not in self.states:
                raise GraphError(
                    f"final_state ({self.final_state}) specified and not found in graph states"
                )
            default_final_state = self.final_state

        elif len(self._start_states) == 1:
            # find the final state in case if a simple sequence of steps
            next_obj = self._start_states[0]
            while next_obj:
                next = next_obj.next
                if not next:
                    default_final_state = next_obj.name
                    break
                next_obj = self[next[0]] if len(next) == 1 else None

        return self._start_states, default_final_state, responders

    def set_flow_source(self, source):
        """set the async flow (storey) source"""
        self._source = source

    def _build_async_flow(self):
        """initialize and build the async/storey DAG"""
        try:
            import storey
        except ImportError:
            raise GraphError("storey package is not installed, use pip install storey")

        def process_step(state, step, root):
            if not state._is_local_function(self.context):
                return
            for item in state.next or []:
                next_state = root[item]
                next_step = step.to(next_state.async_object)
                process_step(next_state, next_step, root)

        for state in self._states.values():
            if hasattr(state, "async_object"):
                if state.kind == StateKinds.queue:
                    if state.path:
                        state._async_object = storey.WriteToV3IOStream(
                            storey.V3ioDriver(), state.path
                        )
                    else:
                        state._async_object = storey.Map(lambda x: x)

                elif not state.async_object or not hasattr(
                    state.async_object, "_outlets"
                ):
                    # if regular class, wrap with storey Map
                    state._async_object = storey.Map(
                        state._handler,
                        full_event=state.full_event,
                        name=state.name,
                        context=self.context,
                    )
                if not state.next and hasattr(state, "responder") and state.responder:
                    # if responder state (return result), add Complete()
                    state.async_object.to(storey.Complete(full_event=True))
                    self._wait_for_result = True

        # todo: allow source array (e.g. data->json loads..)
        source = self._source or storey.Source()
        for next_state in self._start_states:
            next_step = source.to(next_state.async_object)
            process_step(next_state, next_step, self)

        for state in self._states.values():
            # add error handler hooks
            if (state.on_error or self.on_error) and state.async_object:
                error_state = self._states[state.on_error or self.on_error]
                state.async_object.set_recovery_step(error_state.async_object)

        self._controller = source.run()

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

    def is_empty(self):
        """is the graph empty (no child states)"""
        return len(self.states) == 0

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

        next_obj = self._start_states[0]
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


class RootFlowState(FlowState):
    """root flow state"""

    kind = "root"
    _dict_fields = ["states", "engine", "final_state", "on_error"]


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


def _add_graphviz_router(graph, state, source=None, **kwargs):
    if source:
        graph.node("_start", source.name, shape=source.shape, style="filled")
        graph.edge("_start", state.fullname)

    graph.node(state.fullname, label=state.name, shape=state.get_shape())
    for route in state.get_children():
        graph.node(route.fullname, label=route.name, shape=route.get_shape())
        graph.edge(state.fullname, route.fullname)


def _add_graphviz_flow(
    graph, state, source=None, targets=None,
):
    start_states, default_final_state, responders = state.check_and_process_graph(
        allow_empty=True
    )
    graph.node("_start", source.name, shape=source.shape, style="filled")
    for start_state in start_states:
        graph.edge("_start", start_state.fullname)
    for child in state.get_children():
        kind = child.kind
        if kind == StateKinds.router:
            with graph.subgraph(name="cluster_" + child.fullname) as sg:
                _add_graphviz_router(sg, child)
        else:
            graph.node(child.fullname, label=child.name, shape=child.get_shape())
        after = child.after or []
        for item in after:
            previous_object = state[item]
            kw = (
                {"ltail": "cluster_" + previous_object.fullname}
                if previous_object.kind == StateKinds.router
                else {}
            )
            graph.edge(previous_object.fullname, child.fullname, **kw)
        if child.on_error:
            graph.edge(child.fullname, child.on_error, style="dashed")

    # draw targets after the last state (if specified)
    if targets:
        for target in targets or []:
            graph.node(target.fullname, label=target.name, shape=target.get_shape())
            last_state = target.after or default_final_state
            if last_state:
                graph.edge(last_state, target.fullname)


def _generate_graphviz(
    state, renderer, filename=None, format=None, source=None, targets=None, **kw,
):
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            'graphviz is not installed, run "pip install graphviz" first!'
        )
    graph = Digraph("mlrun-flow", format="jpg")
    graph.attr(compound="true", **kw)
    source = source or BaseState("start", shape="egg")
    renderer(graph, state, source=source, targets=targets)
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
        if kind == StateKinds.router:
            server._graph = server._verify_dict(graph, "graph", RouterState)
        elif not kind or kind == StateKinds.root:
            server._graph = server._verify_dict(graph, "graph", RootFlowState)
        else:
            raise GraphError(f"illegal root state {kind}")


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
    """return state object from provided params or classes/objects"""
    if class_name and hasattr(class_name, "to_dict"):
        struct = class_name.to_dict()
        kind = struct.get("kind", StateKinds.task)
        name = name or struct.get("name", struct.get("class_name"))
        cls = classes_map.get(kind, RootFlowState)
        state = cls.from_dict(struct)
        state.function = function
        state.full_event = full_event

    elif class_name and class_name in [">>", "$queue"]:
        if "path" not in class_args:
            raise MLRunInvalidArgumentError(
                "path=<stream path or None> must be specified for queues"
            )
        if not name:
            raise MLRunInvalidArgumentError("queue name must be specified")
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
        state = TaskState(
            class_name,
            class_args,
            handler,
            name=name,
            function=function,
            full_event=full_event,
        )
    else:
        raise MLRunInvalidArgumentError("class_name or handler must be provided")

    if graph_shape:
        state.shape = graph_shape
    return name, state
