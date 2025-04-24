# Copyright 2025 Iguazio
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

from abc import ABC, abstractmethod
from typing import Any, Optional


class AbstractClient(ABC):
    """
    Abstract Base Class for the Kubeflow Pipelines Client.

    This class defines the public interface for interacting with Kubeflow Pipelines.
    """

    @abstractmethod
    def get_kfp_healthz(self):
        """
        Retrieve the healthz status of the KFP API.

        This method retries multiple times until the endpoint responds or a timeout occurs.

        :return: A valid ApiGetHealthzResponse if successful, otherwise None.
        :raises TimeoutError: If the endpoint is not reachable after the specified retries.
        """
        raise NotImplementedError

    @abstractmethod
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        """
        Create a new experiment if it does not already exist.

        This method searches for an experiment by name (and optional namespace), and creates
        it if not found. If found, the existing experiment is returned.

        :param name:         The name of the experiment to create or retrieve.
        :param description:  A description for the experiment.
        :param namespace:    An optional Kubernetes namespace.
        :return: An ApiExperiment object representing the experiment.
        :raises ValueError:  If multiple experiments with the same name are found.
        """
        raise NotImplementedError

    @abstractmethod
    def get_experiment(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        """
        Retrieve an experiment by ID or name.

        This method fetches an experiment using its unique ID or by searching for the name
        (optionally in a specified namespace).

        :param experiment_id:   The ID of the experiment to retrieve.
        :param experiment_name: The name of the experiment to retrieve.
        :param namespace:       An optional Kubernetes namespace for filtering by name.
        :return: An ApiExperiment object representing the experiment.
        :raises ValueError: If neither experiment_id nor experiment_name is provided, or if
                            multiple experiments share the same name.
        """
        raise NotImplementedError

    @abstractmethod
    def run_pipeline(
        self,
        experiment_id: str,
        job_name: str,
        pipeline_package_path: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        pipeline_id: Optional[str] = None,
        version_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        should_enable_caching: Optional[bool] = None,
        service_account: Optional[str] = None,
    ):
        """
        Run a pipeline within a specified experiment.

        This method submits a pipeline run using various optional arguments like a pipeline
        package path, pipeline ID, parameters, and caching settings.

        :param experiment_id:         The ID of the experiment to run the pipeline in.
        :param job_name:              The name to assign to this pipeline run.
        :param pipeline_package_path: An optional path to the pipeline package file (tar.gz, zip, yaml).
        :param params:                An optional dictionary of pipeline parameters.
        :param pipeline_id:           An optional pipeline ID. If provided, the client uses the existing pipeline.
        :param version_id:            An optional pipeline version ID.
        :param pipeline_root:         An optional root path for pipeline outputs.
        :param should_enable_caching:        A flag to enable or disable pipeline caching.
        :param service_account:       An optional Kubernetes service account to run the pipeline.
        :return: An ApiRun object representing the created pipeline run.
        """
        raise NotImplementedError

    @abstractmethod
    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        experiment_id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter_: Optional[str] = None,
    ):
        """
        List pipeline runs with optional filters.

        This method retrieves runs, optionally filtering by experiment ID, namespace, or custom filters.
        Pagination and sorting are also supported.

        :param page_token:    A token for pagination.
        :param page_size:     Number of runs to retrieve per request.
        :param sort_by:       A string specifying how to sort the results.
        :param experiment_id: An optional experiment ID to filter runs by.
        :param namespace:     An optional namespace to filter runs by.
        :param filter_:       A custom filter string (if any).
        :return: An ApiListRunsResponse object containing the runs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str):
        """
        Retrieve details of a specific pipeline run.

        :param run_id: The unique ID of the run to retrieve.
        :return: An ApiRun object with the run details.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_run_completion(
        self,
        run_id: str,
        timeout: int,
        check_interval_seconds: int = 5,
    ):
        """
        Wait for a pipeline run to reach a stable status (e.g., Succeeded or Failed).

        This method polls the run status at a specified interval until it completes
        or a timeout is reached.

        :param run_id:               The unique ID of the run.
        :param timeout:              The total time in seconds to wait for completion.
        :param check_interval_seconds: How often (in seconds) to poll the run status.
        :return: An ApiRun object describing the run at final status.
        :raises TimeoutError: If the run does not complete before the timeout.
        """
        raise NotImplementedError

    @abstractmethod
    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Upload a pipeline package file to Kubeflow Pipelines.

        :param pipeline_package_path: Path to the pipeline package file (zip, tar.gz, yaml, etc.).
        :param pipeline_name:         An optional name to assign to the pipeline.
        :param description:           An optional description for the pipeline.
        :return: An ApiPipeline object representing the uploaded pipeline.
        """
        raise NotImplementedError
