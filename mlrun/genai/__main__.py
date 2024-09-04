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
#
# main file with cli commands using python click library
import importlib.util
import sys

import click
import uvicorn

from mlrun.genai.api import router
from mlrun.genai.chains.base import HistorySaver, SessionLoader
from mlrun.genai.chains.refine import RefineQuery
from mlrun.genai.chains.retrieval import MultiRetriever
from mlrun.genai.config import config, username
from mlrun.genai.workflows import AppServer

default_graph = [
    SessionLoader(),
    RefineQuery(),
    MultiRetriever(),
    HistorySaver(),
]


@click.group()
def cli():
    pass


@click.command()
@click.argument("workflow-name", type=str)
@click.option("-p", "--path", type=str, default=None, help="Path to the workflow file")
@click.option("-r", "--runner", type=str, default="fastapi", help="Runner to use")
@click.option(
    "-t",
    "--workflow-type",
    type=str,
    default="application",
    help="Type of the workflow",
)
def run(
    workflow_name: str,
    runner: str,
    path: str,
    workflow_type: str,
):
    """
    Run workflow application

    :param workflow_name:   The workflow name
    :param runner:          The runner to use, default is fastapi.
    :param path:            The path to the workflow file.
    :param workflow_type:   The type of the workflow. Can be one of mlrun.genai.schemas.WorkflowType

    :return:    None
    """
    # Import the workflow's graph from the path
    if path:
        # Load the module from the given file path
        spec = importlib.util.spec_from_file_location("module_name", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module_name"] = module
        spec.loader.exec_module(module)

        # Retrieve the desired object from the module
        click.echo(f"Using graph from {path}")
        graph = getattr(module, "workflow_graph")
    else:
        # Use the default graph
        click.echo("Using default graph")
        graph = default_graph

    if runner == "nuclio":
        click.echo("Running nuclio is not supported yet")
    elif runner == "fastapi":
        app_server = AppServer()
        app_server.add_workflow(
            project_name="default",
            name=workflow_name,
            graph=graph,
            deployment=config.infer_path(workflow_name),
            workflow_type=workflow_type,
            username=username,
            update=True,
        )
        app = app_server.to_fastapi(router=router)

        # Deploy the fastapi app
        host = config.workflow_deployment["host"]
        port = config.workflow_deployment["port"]
        click.echo(f"Running workflow {workflow_name} with fastapi on {host}")
        uvicorn.run(app, host=host, port=port)

    else:
        click.echo(
            f"Runner {runner} not supported. Supported runners are: nuclio, fastapi"
        )


cli.add_command(run)


if __name__ == "__main__":
    cli()
