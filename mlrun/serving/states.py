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
from copy import deepcopy

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from ..model import ModelObj, ObjectDict
from ..utils import create_class


class StateKinds:
    router = "router"
    task = "task"


_task_state_fields = ["kind", "class_name", "class_args", "handler"]


def new_model_endpoint(class_name, model_path, handler=None, **class_args):
    class_args = deepcopy(class_args)
    class_args["model_path"] = model_path
    return ServingTaskState(class_name, class_args, handler=handler)


def new_remote_endpoint(url, **class_args):
    class_args = deepcopy(class_args)
    class_args["url"] = url
    return ServingTaskState("$remote", class_args)


class ServingTaskState(ModelObj):
    kind = "task"
    _dict_fields = _task_state_fields
    _default_class = ""

    def __init__(self, class_name=None, class_args=None, handler=None, name=None):
        self.name = name
        self.class_name = class_name
        self.class_args = class_args or {}
        self.handler = handler
        self._handler = None
        self._object = None
        self.context = None
        self._class_object = None

    def init_object(self, context, namespace, mode="sync"):
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

        if not self._object:
            print(self.class_args)
            self._object = self._class_object(context, self.name, **self.class_args)
            self._handler = getattr(self._object, self.handler or "do_event")

        if mode != "skip":
            self._post_init(mode)

    @property
    def object(self):
        return self._object

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(mode)

    def run(self, event, *args, **kwargs):
        return self._handler(event, *args, **kwargs)


class ServingRouterState(ServingTaskState):
    kind = "router"
    _dict_fields = _task_state_fields + ["routes"]
    _default_class = "mlrun.serving.ModelRouter"

    def __init__(
        self, class_name=None, class_args=None, handler=None, routes=None, name=None
    ):
        super().__init__(class_name, class_args, handler, name=name)
        self._routes = {}
        self.routes = routes

    @property
    def routes(self):
        return self._routes

    @routes.setter
    def routes(self, routes: dict):
        self._routes = ObjectDict.from_dict(classes_map, routes, "task")

    def add_route(self, key, route):
        self._routes[key] = route
        return route

    def clear_routes(self, routes: list):
        if not routes:
            routes = self._routes.keys()
        for key in routes:
            del self._routes[key]

    def init_object(self, context, namespace, mode="sync"):
        self.class_args = self.class_args or {}
        self.class_args["routes"] = self._routes
        super().init_object(context, namespace, "skip")
        del self.class_args["routes"]

        for route in self._routes.values():
            route.init_object(context, namespace, mode)

        self._post_init(mode)

    def __getitem__(self, name):
        return self._routes[name]


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
        class_object = namespace[class_name]
        return class_object

    try:
        class_object = create_class(class_name)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, class {class_name} not found, {e}")
    return class_object


classes_map = {"task": ServingTaskState}
