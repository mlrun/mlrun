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

from typing import Any, Optional

import mlrun
import mlrun.errors
from mlrun.utils import logger


class MarketplaceAgent:
    """
    This class provides methods to get information about an agent and deploy it
    as an MLRun application runtime. It automatically optimizes the build process
    by caching base images with requirements and reusing them across deployments.
    """

    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        kind: str,
        protocol: str,
        framework: str,
        asset_url: str,
        requirements: list[str],
        default_base_image: str,
        default_port: int,
        default_command: str,
        default_args: list[str],
        inputs: list[dict[str, Any]],
        categories: Optional[list[str]] = None,
        default_workdir: Optional[str] = None,
        build_extra: Optional[str] = None,
    ):
        """
        :param name: Agent name
        :param version: Agent version
        :param author: Agent author
        :param description: Agent description
        :param kind: Agent kind (e.g., "atomic-agent")
        :param protocol: Communication protocol (e.g., "A2A")
        :param framework: Implementation framework (e.g., "LangChain")
        :param asset_url: URL to agent code ZIP/archive
        :param requirements: Python requirements list
        :param default_base_image: Default Docker base image
        :param default_port: Default application port
        :param default_command: Default command to run
        :param default_args: Default command arguments
        :param inputs: List of input configurations (secrets/env vars)
        :param categories: Optional list of categories
        :param default_workdir: Optional default working directory for build
        :param build_extra: Optional raw Dockerfile commands for build
        """
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.kind = kind
        self.protocol = protocol
        self.framework = framework
        self.asset_url = asset_url
        self.requirements = requirements
        self.default_base_image = default_base_image
        self.default_port = default_port
        self.default_command = default_command
        self.default_args = default_args
        self.inputs = inputs
        self.categories = categories or []
        self.default_workdir = default_workdir
        self.build_extra = build_extra

        # Extract mandatory configurations from inputs
        # Mandatory = required:true AND no default value
        self.mandatory_configurations = [
            inp["name"]
            for inp in self.inputs
            if inp.get("required", False) and not inp.get("default")
        ]

    def info(self) -> None:
        """
        Get information about the agent.

        :return: Formatted string with agent information
        """
        inputs_keys = [inp["name"] for inp in self.inputs]
        optional_inputs = [
            k for k in inputs_keys if k not in self.mandatory_configurations
        ]

        info_str = (
            f"Agent: {self.name}\n"
            f"Version: {self.version}\n"
            f"Author: {self.author}\n"
            f"Kind: {self.kind}\n"
            f"Description: {self.description}\n"
            f"Framework: {self.framework}\n"
            f"Protocol: {self.protocol}\n"
            f"Required Inputs: "
            f"{', '.join(self.mandatory_configurations)}\n"
            f"Optional Inputs: {', '.join(optional_inputs)}"
        )
        print(info_str)

    def _validate_mandatory_configs(
        self, kwargs: dict[str, Any], mandatory_configs: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Validate that all mandatory configurations are provided and apply defaults.

        Separates configs into environment variables and secrets based on input type.
        Mandatory configs (required=true, no default) must be provided by user.
        Optional configs (required=false or has default) use provided value or default.

        :param kwargs: User-provided configurations
        :param mandatory_configs: List of mandatory configuration keys
        :return: Tuple of (env_vars, secrets) dictionaries
        :raises MLRunInvalidArgumentError: If mandatory configs are missing
        """
        env_vars = {}
        secrets = {}
        missing_configs = []

        # Check mandatory configs (must be provided)
        for config_key in mandatory_configs:
            if config_key not in kwargs:
                missing_configs.append(config_key)

        if missing_configs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Missing mandatory configurations: {', '.join(missing_configs)}"
            )

        # Process all inputs (apply defaults for optional ones) and separate by type
        for inp in self.inputs:
            key = inp["name"]
            value = None

            if key in kwargs:
                # User provided value
                value = kwargs[key]
            elif inp.get("default"):
                # Use default value if not provided and default exists
                value = inp["default"]

            # Store in appropriate dict based on type
            if value is not None:
                if inp.get("type") == "secret":
                    secrets[key] = value
                else:
                    env_vars[key] = value

        return env_vars, secrets

    def _get_build_extra_commands(self) -> Optional[str]:
        """
        Generate build extra Dockerfile commands from agent asset configuration.

        Combines default_workdir (if set) and build_extra (if set) into a single
        Dockerfile commands string.

        :return: Dockerfile commands string or None if no extra commands
        """
        commands = []

        # Add WORKDIR if specified
        if self.default_workdir:
            commands.append(f"WORKDIR {self.default_workdir}")

        # Add any additional build extra commands
        if self.build_extra:
            commands.append(self.build_extra.rstrip())

        return "\n".join(commands) + "\n" if commands else None

    def deploy(
        self,
        project: str,
        source_url: Optional[
            str
        ] = None,  # todo: delete when there is backend (request source from BE inside this function)
        gateway_config: Optional[dict[str, Any]] = None,
        force_rebuild: bool = False,
        **kwargs,
    ):
        """
        Deploy the marketplace agent as an MLRun application runtime.

        Builds and deploys the agent with requirements and source code.
        On subsequent deployments, reuses cached built image to skip rebuild
        (unless force_rebuild=True).

        :param project: MLRun project name
        :param gateway_config: API gateway configuration dict. If provided,
            creates an API gateway with these settings. Supports:
            - name: Gateway name (default: "{agent_name}-gateway")
            - path: URL path (default: "/")
            - authentication_mode: Auth mode (e.g., "none", "accessKey")
            - authentication_creds: Auth credentials
            - direct_port_access: Enable direct port access (default: False)
            - ssl_redirect: Enable SSL redirect (default: True)
            - set_as_default: Set as default gateway (default: False)
        :param force_rebuild: Force rebuild even if cached image exists (default: False)
        :param kwargs: Additional configuration options including:
            - base_image: Override default base image (for initial build)
            - port: Override default port
            - command: Override default command
            - args: Override default args
            - requirements: Override default requirements list or file path
            - create_default_api_gateway: Whether to create default API gateway
                (default: False, ignored if gateway_config is provided)
            - Any input configurations as specified in agent's inputs:
                - Inputs with type="secret" are stored in project secrets
                  and set as environment variables
                - Inputs with type="env" are set as regular environment variables
                - Note: All functions in a project share the same secrets
        :return: Deployment URL for invoking the agent
        """
        # Validate mandatory configurations and separate into env vars and secrets
        env_vars, secrets = self._validate_mandatory_configs(
            kwargs, self.mandatory_configurations
        )

        # Get or create project
        project_obj = mlrun.get_or_create_project(project)

        # Determine base image: user override > agent default > None (runtime uses its default)
        base_image = kwargs.get("base_image") or self.default_base_image

        # Try to reuse existing function's built image to avoid rebuilding
        use_cached_image = False
        cached_image = None
        if not force_rebuild:
            try:
                existing_func = project_obj.get_function(self.name)

                # For application runtime, the built image is in the sidecar config
                sidecars = existing_func.spec.config.get("spec.sidecars", [])
                if sidecars and len(sidecars) > 0:
                    existing_image = sidecars[0].get("image")
                    # Check if it's a built image (not empty and not the base image)
                    if existing_image and existing_image != base_image:
                        use_cached_image = True
                        cached_image = existing_image
                        logger.info(
                            "Reusing cached base image from previous deployment",
                            agent=self.name,
                            cached_image=cached_image,
                        )
            except Exception:
                # Function doesn't exist or error loading it - will build new
                pass

        # Set up application function
        if use_cached_image:
            # Use cached built image - skip requirements to avoid rebuild
            app = project_obj.set_function(
                kind="application",
                image=cached_image,
                name=self.name,
            )
        else:
            # First deploy or no cache - build with requirements
            app = project_obj.set_function(
                kind="application",
                image=base_image,  # None is valid - runtime will use its default
                name=self.name,
            )

            # Add requirements (only on first build)
            reqs = kwargs.get("requirements")
            if reqs is not None:
                # User provided override
                if isinstance(reqs, str):
                    app.with_requirements(requirements_file=reqs)
                else:
                    app.with_requirements(requirements=reqs)
            elif self.requirements:
                # Use default from agent asset
                app.with_requirements(requirements=self.requirements)

            # Add build extra commands (WORKDIR, etc.) - only on first build
            build_extra_commands = self._get_build_extra_commands()
            if build_extra_commands:
                app.spec.build.extra = build_extra_commands

        # todo: request source from the backend when implemented
        # Always add source archive (loaded at runtime via store:// URI)
        app.with_source_archive(source=source_url, pull_at_runtime=False)

        # Configure application port
        app.set_internal_application_port(kwargs.get("port") or self.default_port)

        # Configure command and args
        app.spec.command = kwargs.get("command") or self.default_command
        app.spec.args = kwargs.get("args") or self.default_args

        # Store secrets in project secrets
        # MLRun automatically mounts project secrets to pods
        # Assumption: All functions in a project share the same secrets (no collision)
        if secrets:
            # Store secrets in project secret store
            project_obj.set_secrets(secrets)
            # Secrets will be automatically available as env vars in the pod

        # Set regular environment variables
        for key, value in env_vars.items():
            app.set_env(key, value)

        # Deploy application
        # If gateway_config provided, don't create default gateway (we'll create custom one)
        create_default_gateway = (
            kwargs.get("create_default_api_gateway", False)
            if not gateway_config
            else False
        )
        app.deploy(
            with_mlrun=False,
            create_default_api_gateway=create_default_gateway,
            show_on_failure=True,
        )

        # Create API gateway if config is provided
        if gateway_config:
            # Set default gateway name if not provided
            gateway_params = gateway_config.copy()
            if "name" not in gateway_params:
                gateway_params["name"] = f"{self.name}-gateway"

            app.create_api_gateway(**gateway_params)

        # Get the deployment URL
        deployment_url = app.status.url if hasattr(app.status, "url") else None
        if not deployment_url and hasattr(app.status, "external_invocation_urls"):
            # Fallback to first external URL
            urls = app.status.external_invocation_urls
            deployment_url = urls[0] if urls else None

        logger.info(
            "Agent deployed successfully",
            agent=self.name,
            project=project,
            url=deployment_url,
        )

        return deployment_url


# todo: get the metadata from the backend using name once it's implemented
def import_agent(name: str, agent_metadata: dict) -> MarketplaceAgent:
    """
    Import an agent from the MLRun marketplace.

    :param name: Agent name (e.g., "marketplace://atomic-writer:0.0.1")
    :param agent_metadata: Agent metadata dict containing all configuration
    :return: MarketplaceAgent instance

    Example:
         agent = mlrun.import_agent("marketplace://atomic-writer:0.0.1", agent_metadata)
         agent.info()
         agent.deploy(project="my-project", OPENAI_API_KEY="sk-...", ...)
    """
    # agent_metadata = MarketplaceBackend.get_asset_metadata(name) # todo: wire with actual backend call to get metadata
    return MarketplaceAgent(**agent_metadata)


def deploy_agent(
    name: str,
    project: str,
    agent_metadata: dict,  # todo: delete when there is backend to get it from
    source: Optional[str] = None,  # todo: delete when there is backend to get it from
    gateway_config: Optional[dict[str, Any]] = None,
    force_rebuild: bool = False,
    **kwargs,
):
    """
    Deploy agent frm the marketplace as an MLRun application runtime in a single step.

    :param name: Agent name (e.g., "marketplace://atomic-writer:0.0.1")
    :param project: MLRun project name
    :param gateway_config: API gateway configuration dict
        (see MarketplaceAgent.deploy for details)
    :param force_rebuild: Force rebuild even if cached image exists (default: False)
    :param kwargs: Additional configuration options including:
        - base_image: Override default base image (e.g., "ubuntu:22.04")
        - requirements: Override requirements (list or file path)
        - Any input configurations as specified in agent's inputs:
            - Inputs with type="secret" are stored in project secrets
            - Inputs with type="env" are set as regular environment variables
            - Note: All functions in a project share the same secrets
        (see MarketplaceAgentDeployer.deploy for full options)
    :return: Deployment URL for invoking the agent

    Example:
        mlrun.deploy_agent(
            "marketplace://atomic-writer:0.0.1",
             project="my-project",
             gateway_config={
                "authentication_mode": "none",
                "path": "/",
                 "ssl_redirect": True,
             },
             OPENAI_API_KEY="sk-...",  # Stored securely as a secret
         )
    """
    mp_agent = import_agent(
        name, agent_metadata
    )  # todo: remove metadata parameter when there is backend
    return mp_agent.deploy(
        project=project,
        source_url=source,  # todo: delete when there is backend (the source will be requested from thr BE by deploy())
        gateway_config=gateway_config,
        force_rebuild=force_rebuild,
        **kwargs,
    )
