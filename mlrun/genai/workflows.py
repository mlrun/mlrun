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

import mlrun
from mlrun import serving
from mlrun.genai.client import client
from mlrun.genai.config import config as default_config
from mlrun.genai.schemas import APIDictResponse
from mlrun.genai.sessions import get_session_store
from mlrun.utils import get_caller_globals


class AppServer:
    def __init__(self, config=None, verbose=False):
        self._config = config or default_config
        self._session_store = get_session_store(self._config)
        self._workflows = {}
        self.verbose = verbose

    def set_config(self, config):
        self._config = config
        self._session_store = get_session_store(self._config)
        for workflow in self._workflows.values():
            workflow._server = None

    def add_workflow(
        self,
        project_name: str,
        name: str,
        graph: list = None,
        deployment: str = None,
        update: bool = False,
        workflow_type: str = None,
        username: str = None,
    ):
        # Check if workflow already exists:
        if name in self._workflows:
            raise ValueError(f"workflow {name} already exists")
        workflow = client.get_workflow(project_name=project_name, workflow_name=name)
        if workflow:
            if not update:
                raise ValueError(
                    f"workflow {name} already exists, to update set update=True"
                )
            else:
                # Update workflow:
                if graph:
                    workflow.add_graph(graph)
                if deployment:
                    workflow.deployment = deployment
                workflow = client.update_workflow(
                    project_name=project_name, workflow=workflow
                )
        else:
            # Workflow does not exist, create it:
            owner_id = client.get_user(username=username)["uid"]
            workflow = {
                "name": name,
                "deployment": deployment,
                "graph": graph,
                "workflow_type": workflow_type,
                "owner_id": owner_id,
            }
            # Add workflow to database:
            workflow = client.create_workflow(
                project_name=project_name,
                workflow=workflow,
            )
        # Add workflow to app server:
        self._workflows[name] = {
            "uid": workflow.uid,
            "project_name": project_name,
        }
        return workflow

    def add_workflows(self, project_name: str, workflows: dict):
        for name, workflow in workflows.items():
            self.add_workflow(project_name=project_name, **workflow)

    def get_workflow(self, name):
        workflow = self._workflows.get(name)
        uid = workflow.get("uid")
        project_name = workflow.get("project_name")
        return client.get_workflow(project_name=project_name, workflow_id=uid)

    def run_workflow(self, name, event):
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"workflow {name} not found")
        app_workflow = AppWorkflow(self, name=workflow.name, graph=workflow.graph)
        return app_workflow.run(event)

    def api_startup(self):
        print("\nstartup event\n")

    def to_fastapi(self, router=None):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI()

        # Add CORS middleware, remove in production
        origins = ["*"]  # React app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        extra = app.extra or {}
        extra["app_server"] = self
        app.extra = extra
        if router:
            app.include_router(router)
        return app


app_server = AppServer()

# workflows cache
workflows = {}


class AppWorkflow:
    def __init__(self, parent, name=None, graph=None):
        self.name = name or ""
        self._parent = parent
        self._graph = None
        self._server = None

        if graph:
            self.graph = graph

    @property
    def graph(self) -> serving.states.RootFlowStep:
        return self._graph

    @graph.setter
    def graph(self, graph):
        if isinstance(graph, list):
            if not graph:
                raise ValueError("graph list must not be empty")
            graph_obj = mlrun.serving.states.RootFlowStep()
            step = graph_obj
            for item in graph:
                if isinstance(item, dict):
                    step = step.to(**item)
                else:
                    step = step.to(item)
            step.respond()
            self._graph = graph_obj
            return

        if isinstance(graph, dict):
            graph = mlrun.serving.states.RootFlowStep.from_dict(graph)
        self._graph = graph

    def get_server(self):
        if self._server is None:
            namespace = get_caller_globals()
            server = serving.create_graph_server(
                graph=self.graph,
                parameters={},
                verbose=self._parent.verbose or True,
                graph_initializer=self.lc_initializer,
            )
            server.init_states(context=None, namespace=namespace)
            server.init_object(namespace)
            self._server = server
            return server
        return self._server

    def lc_initializer(self, server):
        context = server.context

        def register_prompt(
            name, template, description: str = None, llm_args: dict = None
        ):
            if not hasattr(context, "prompts"):
                context.prompts = {}
            context.prompts[name] = (template, llm_args)

        if getattr(context, "_config", None) is None:
            context._config = self._parent._config
        if getattr(context, "session_store", None) is None:
            context.session_store = self._parent._session_store

    def run(self, event, db_session=None):
        # todo: pass sql db_session to steps via context or event
        server = self.get_server()
        try:
            resp = server.test("", body=event)
        except Exception as e:
            server.wait_for_completion()
            raise e

        return APIDictResponse(
            success=True,
            data={
                "answer": resp.results["answer"],
                "sources": resp.results["sources"],
                "returned_state": {},
            },
        )
