import mlrun
from mlrun import serving
from mlrun.utils import get_caller_globals

from .config import config as default_config
from .schema import ApiDictResponse
from .sessions import get_session_store


class AppServer:

    def __init__(self, config=None, verbose=False):
        self._config = config or default_config
        self._session_store = get_session_store(self._config)
        self._pipelines = {}
        self.verbose = verbose

    def set_config(self, config):
        self._config = config
        self._session_store = get_session_store(self._config)
        for pipeline in self._pipelines.values():
            pipeline._server = None

    def add_pipeline(self, name, graph):
        pipeline = AppPipeline(self, name, graph)
        self._pipelines[name] = pipeline
        return pipeline

    def add_pipelines(self, pipelines: dict):
        for name, graph in pipelines.items():
            self.add_pipeline(name, graph)

    def get_pipeline(self, name):
        return self._pipelines.get(name)

    def run_pipeline(self, name, event):
        pipeline = self.get_pipeline(name)
        if not pipeline:
            raise ValueError(f"pipeline {name} not found")
        return pipeline.run(event)

    def api_startup(self):
        print("\nstartup event\n")

    def to_fastapi(self, router=None, with_controller=False):
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

        if not router:
            if with_controller:
                from llmapps.controller.api import router as controller_router

                router = controller_router
            else:
                from llmapps.app.api import base_router

                router = base_router

        extra = app.extra or {}
        extra["app_server"] = self
        app.extra = extra
        router.add_event_handler("startup", self.api_startup)

        app.include_router(router)
        return app


app_server = AppServer()

# pipelines cache
pipelines = {}


class AppPipeline:
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

        print("resp: ", resp)
        return ApiDictResponse(
            success=True,
            data={
                "answer": resp.results["answer"],
                "sources": resp.results["sources"],
                "returned_state": {},
            },
        )
