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
import json
import pathlib
from copy import deepcopy

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from ..model import ModelObj, ObjectDict
from ..utils import create_class, create_function

callable_prefix = "_"


class StateKinds:
    router = "router"
    task = "task"
    flow = "flow"


_task_state_fields = [
    "kind",
    "class_name",
    "class_args",
    "handler",
    "skip_context",
    "next",
    "resource",
    "comment",
    "end",
    "full_event",
]


def new_model_endpoint(class_name, model_path, handler=None, **class_args):
    class_args = deepcopy(class_args)
    class_args["model_path"] = model_path
    return ServingTaskState(class_name, class_args, handler=handler)


def new_remote_endpoint(url, **class_args):
    class_args = deepcopy(class_args)
    class_args["url"] = url
    return ServingTaskState("$remote", class_args)


class BaseState(ModelObj):
    kind = "BaseState"
    shape = "ellipse"
    _dict_fields = ["kind", "comment", "next", "end", "resource"]

    def __init__(self, name=None, next=None):
        self.name = name
        self._parent = None
        self.comment = None
        self.context = None
        self.next = next
        self.end = None
        self.resource = None

    def set_parent(self, parent):
        self._parent = parent

    def set_next(self, key):
        if not self.next:
            self.next = [key]
        elif key not in self.next:
            self.next.append(key)

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context

    def get_children(self):
        return []

    def __iter__(self):
        yield from []

    @property
    def fullname(self):
        name = self.name or ""
        if self._parent and self._parent.fullname:
            name = ".".join([self._parent.fullname, name])
        return name.replace(":", "_")  # replace for graphviz escaping

    def _post_init(self, mode="sync"):
        pass

    def path_to_state(self, path):
        path = path or ""
        tree = path.split(".")
        next_obj = self
        for state in tree:
            if state not in next_obj:
                raise ValueError(
                    f"step {state} doesnt exist in the graph under {next_obj.fullname}"
                )
            next_obj = next_obj[state]
        return next_obj


class ServingTaskState(BaseState):
    kind = "task"
    _dict_fields = _task_state_fields
    _default_class = ""

    def __init__(
        self,
        class_name=None,
        class_args=None,
        handler=None,
        name=None,
        next=None,
        full_event=None,
    ):
        super().__init__(name, next)
        self.class_name = class_name
        self.class_args = class_args or {}
        self.handler = handler
        self._handler = None
        self._object = None
        self.skip_context = None
        self.context = None
        self._class_object = None
        self.full_event = full_event

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        if isinstance(self.class_name, type):
            self._class_object = self.class_name
            self.class_name = self.class_name.__name__

        self.context = context
        if not self._class_object:
            if self.class_name == "$remote":
                self._class_object = RemoteHttpHandler
            else:
                self._class_object = get_class(
                    self.class_name or self._default_class, namespace
                )

        if not self._object or reset:
            class_args = {}
            for key, arg in self.class_args.items():
                if key.startswith(callable_prefix):
                    class_args[key[1:]] = get_function(arg, namespace)
                else:
                    class_args[key] = arg
            class_args.update(extra_kwargs)
            if self.skip_context is None or not self.skip_context:
                class_args["name"] = self.name
                class_args["context"] = self.context
            self._object = self._class_object(**class_args)

            handler = self.handler
            if handler:
                if not hasattr(self._object, handler):
                    raise ValueError(
                        f"handler ({handler}) specified but doesnt exist in class"
                    )
            else:
                if hasattr(self._object, "do"):
                    handler = "do"
                elif hasattr(self._object, "do_event"):
                    handler = "do_event"
                    self.full_event = True
            if handler:
                self._handler = getattr(self._object, handler, None)

        if mode != "skip":
            self._post_init(mode)

    @property
    def object(self):
        return self._object

    def clear_object(self):
        self._object = None

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(mode)

    def run(self, event, *args, **kwargs):
        if self.full_event:
            return self._handler(event, *args, **kwargs)
        event.body = self._handler(event.body, *args, **kwargs)
        return event


class ServingRouterState(ServingTaskState):
    kind = "router"
    shape = "doubleoctagon"
    _dict_fields = _task_state_fields + ["routes"]
    _default_class = "mlrun.serving.ModelRouter"

    def __init__(
        self, class_name=None, class_args=None, handler=None, routes=None, name=None
    ):
        super().__init__(class_name, class_args, handler, name=name)
        self._routes = {}
        self.routes = routes

    def get_children(self):
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
            route = ServingTaskState(class_name, class_args, handler=handler)
        route = self._routes.update(key, route)
        route.set_parent(self)
        return route

    def clear_children(self, routes: list):
        if not routes:
            routes = self._routes.keys()
        for key in routes:
            del self._routes[key]

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.class_args = self.class_args or {}
        super().init_object(
            context, namespace, "skip", reset=reset, routes=self._routes, **extra_kwargs
        )

        for route in self._routes.values():
            route.set_parent(self)
            route.init_object(context, namespace, mode, reset=reset)

        self._post_init(mode)

    def __getitem__(self, name):
        return self._routes[name]

    def __setitem__(self, name, route):
        self.add_route(name, route)

    def __delitem__(self, key):
        del self._routes[key]

    def __iter__(self):
        yield from self._routes.keys()

    def plot(self, filename=None, format=None):
        return _generate_graphviz(self, _add_gviz_router, filename, format)


class ServingFlowState(BaseState):
    kind = "flow"
    _dict_fields = BaseState._dict_fields + ["states", "start_at"]

    def __init__(self, name=None, states=None, next=None, start_at=None):
        super().__init__(name, next)
        self._states = None
        self.states = states
        self.start_at = start_at
        self.from_state = None
        self._last_added = None

    def get_children(self):
        return self._states.values()

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        self._states = ObjectDict.from_dict(classes_map, states, "task")

    def add_state(
        self,
        key,
        state=None,
        class_name=None,
        class_args=None,
        handler=None,
        after=None,
    ):
        """add child route state or class to the router

        :param key:        unique name (and path) for the child state
        :param state:      child state object (Task, ..)
        :param class_name: class name to build the state from (when state is not provided)
        :param class_args: class init arguments
        :param handler:    class handler to invoke on run/event
        """

        if not state and not class_name:
            raise ValueError("state or class_name must be specified")
        if not state:
            state = ServingTaskState(class_name, class_args, handler=handler)
        state = self._states.update(key, state)
        state.set_parent(self)

        if not self.start_at and len(self._states) <= 1:
            self.start_at = key

        if after:
            if isinstance(after, str):
                if after not in self._states.keys():
                    raise ValueError(
                        f"there is no state named {after}, cant set next state"
                    )
                after = self._states[after]
            after.set_next(key)
        elif self._last_added:
            self._last_added.set_next(key)
        self._last_added = state
        return state

    def clear_children(self, states: list):
        if not states:
            states = self._states.keys()
        for key in states:
            del self._states[key]

    def __getitem__(self, name):
        return self._states[name]

    def __setitem__(self, name, state):
        self.add_state(name, state)

    def __delitem__(self, key):
        del self._states[key]

    def __iter__(self):
        yield from self._states.keys()

    def init_object(self, context, namespace, mode="sync", reset=False, **extra_kwargs):
        self.context = context
        for state in self._states.values():
            state.set_parent(self)
            state.init_object(context, namespace, mode, reset=reset)
        self._post_init(mode)

    def get_start_state(self, from_state=None):
        from_state = from_state or self.from_state or self.start_at
        if not from_state:
            raise ValueError(
                f"start step {from_state} was not specified in {self.name}"
            )
        return self.path_to_state(from_state)

    def run(self, event, *args, **kwargs):
        next_obj = self.get_start_state(kwargs.get("from_state", None))
        while next_obj:
            event = next_obj.run(event, *args, **kwargs)
            next = next_obj.next
            next_obj = self[next[0]] if next else None
        return event

    def plot(self, filename=None, format=None):
        return _generate_graphviz(self, _add_gviz_flow, filename, format)


class ServingRootFlowState(ServingFlowState):
    kind = "rootFlow"
    _dict_fields = ["states", "start_at"]


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


def get_class(class_name, namespace):
    """return class object from class name string"""
    if isinstance(class_name, type):
        return class_name
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
    if function in namespace:
        return namespace[function]

    try:
        function_object = create_function(function)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, function {function} not found, {e}")
    return function_object


classes_map = {
    "task": ServingTaskState,
    "router": ServingRouterState,
    "flow": ServingFlowState,
}


def _add_gviz_router(g, state):
    g.node(state.fullname, label=state.name, shape=state.shape)
    for route in state.get_children():
        g.node(route.fullname, label=route.name, shape=route.shape)
        g.edge(state.fullname, route.fullname)


def _add_gviz_flow(g, state):
    for child in state.get_children():
        kind = child.kind
        if kind == StateKinds.router:
            with g.subgraph(name="cluster_" + child.fullname) as sg:
                _add_gviz_router(sg, child)
        else:
            g.node(child.fullname, label=child.name, shape=child.shape)
        next = child.next or []
        for item in next:
            next_object = state[item]
            kw = (
                {"ltail": "cluster_" + child.fullname}
                if child.kind == StateKinds.router
                else {}
            )
            g.edge(child.fullname, next_object.fullname, **kw)


def _generate_graphviz(state, renderer, filename=None, format=None):
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            'graphviz is not installed, run "pip install graphviz" first!'
        )
    g = Digraph("mlrun-flow", format="jpg")
    g.attr(compound="true")
    renderer(g, state)
    if filename:
        suffix = pathlib.Path(filename).suffix
        if suffix:
            filename = filename[: -len(suffix)]
            format = format or suffix[1:]
        format = format or "png"
        g.render(filename, format=format)
    return g
