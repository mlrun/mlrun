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
from typing import Optional

import kfp_server_api


class AbstractClient(ABC):
    """
    Abstract Base Class for the Kubeflow Pipelines Client.
    Defines the public interface for interacting with Kubeflow Pipelines.
    """

    @abstractmethod
    def get_kfp_healthz(self) -> kfp_server_api.ApiGetHealthzResponse:
        """
        Retrieve the healthz information of the KFP deployment.

        Returns:
            An ApiGetHealthzResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.ApiExperiment:
        """
        Create (or retrieve) an experiment.

        Args:
            name: Name of the experiment.
            description: Optional description.
            namespace: Optional Kubernetes namespace.

        Returns:
            An ApiExperiment object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_experiment(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.ApiExperiment:
        """
        Get details of an experiment.

        Args:
            experiment_id: Experiment ID.
            experiment_name: Experiment name.
            namespace: Optional Kubernetes namespace.

        Returns:
            An ApiExperiment object.
        """
        raise NotImplementedError

    @abstractmethod
    def run_pipeline(
        self,
        experiment_id: str,
        job_name: str,
        pipeline_package_path: Optional[str] = None,
        params: Optional[dict] = None,
        pipeline_id: Optional[str] = None,
        version_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        service_account: Optional[str] = None,
    ) -> kfp_server_api.ApiRun:
        """
        Run a specified pipeline.

        Args:
            experiment_id: Experiment ID.
            job_name: Name of the job.
            pipeline_package_path: Local path to the pipeline package.
            params: Pipeline parameters.
            pipeline_id: Pipeline ID.
            version_id: Pipeline version ID.
            pipeline_root: Pipeline output root.
            enable_caching: Flag to enable caching.
            service_account: Kubernetes service account.

        Returns:
            An ApiRun object.
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
        filter: Optional[str] = None,
    ) -> kfp_server_api.ApiListRunsResponse:
        """
        List pipeline runs.

        Args:
            page_token: Token for pagination.
            page_size: Number of runs per page.
            sort_by: Sorting order.
            experiment_id: Optional experiment ID filter.
            namespace: Optional namespace filter.
            filter: Optional filter string.

        Returns:
            An ApiListRunsResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_recurring_run(self, job_id: str) -> kfp_server_api.ApiJob:
        """
        Get details of a recurring run.

        Args:
            job_id: Job ID.

        Returns:
            An ApiJob object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str) -> kfp_server_api.ApiRun:
        """
        Get details of a pipeline run.

        Args:
            run_id: Run ID.

        Returns:
            An ApiRun object.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_run_completion(
        self, run_id: str, timeout: int
    ) -> kfp_server_api.ApiRun:
        """
        Wait for a run to complete.

        Args:
            run_id: Run ID.

            timeout: Timeout in seconds.

        Returns:
            An ApiRun object.
        """
        raise NotImplementedError

    @abstractmethod
    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> kfp_server_api.ApiPipeline:
        """
        Upload a pipeline package.

        Args:
            pipeline_package_path: Local path to the pipeline package.
            pipeline_name: Optional pipeline name.
            description: Optional description.

        Returns:
            An ApiPipeline object.
        """
        raise NotImplementedError
