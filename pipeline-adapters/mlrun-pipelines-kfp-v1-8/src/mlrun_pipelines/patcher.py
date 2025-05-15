# Copyright 2024 Iguazio
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

import typing

import mlrun
from mlrun.errors import err_to_str
from mlrun.utils import logger
from mlrun_pipelines.imports import dsl, kfp


# When we run pipelines, the kfp.compile.Compile.compile() method takes the decorated function with @dsl.pipeline and
# converts it to a k8s object. As part of the flow in the Compile.compile() method,
# we call _create_and_write_workflow, which builds a dictionary from the workflow and then writes it to a file.
# Unfortunately, the kfp sdk does not provide an API for configuring priority_class_name and other attributes.
# I ran across the following problem when seeking for a method to set the priority_class_name:
# https://github.com/kubeflow/pipelines/issues/3594
# When we patch the _create_and_write_workflow, we can eventually obtain the dictionary right before we write it
# to a file and enrich it with argo compatible fields, make sure you looking for the same argo version we use
# https://github.com/argoproj/argo-workflows/blob/release-2.7/pkg/apis/workflow/v1alpha1/workflow_types.go
def _create_enriched_mlrun_workflow(
    self,
    pipeline_func: typing.Callable,
    pipeline_name: typing.Optional[str] = None,
    pipeline_description: typing.Optional[str] = None,
    params_list: typing.Optional[list[dsl.PipelineParam]] = None,
    pipeline_conf: typing.Optional[dsl.PipelineConf] = None,
):
    """Call internal implementation of create_workflow and enrich with mlrun functions attributes"""
    from mlrun import pipeline_context
    from mlrun.projects.pipelines import (
        _enrich_kfp_pod_security_context,
        _set_function_attribute_on_kfp_pod,
    )

    workflow = self._original_create_workflow(
        pipeline_func, pipeline_name, pipeline_description, params_list, pipeline_conf
    )
    # We don't want to interrupt the original flow and don't know all the scenarios the function could be called.
    # that's why we have try/except on all the code of the enrichment and also specific try/except for errors that
    # we know can be raised.
    try:
        functions = []
        if pipeline_context.functions:
            try:
                functions = pipeline_context.functions.values()
            except Exception as err:
                logger.debug(
                    "Unable to retrieve project functions, not enriching workflow with mlrun",
                    error=err_to_str(err),
                )
                return workflow

        # enrich each pipeline step with your desire k8s attribute
        for kfp_step_template in workflow["spec"]["templates"]:
            if kfp_step_template.get("container"):
                for function_obj in functions:
                    # we condition within each function since the comparison between the function and
                    # the kfp pod may change depending on the attribute type.
                    _set_function_attribute_on_kfp_pod(
                        kfp_step_template,
                        function_obj,
                        "PriorityClassName",
                        "priority_class_name",
                    )
                    _enrich_kfp_pod_security_context(
                        kfp_step_template,
                        function_obj,
                    )
    except mlrun.errors.MLRunInvalidArgumentError:
        raise
    except Exception as err:
        logger.debug(
            "Something in the enrichment of kfp pods failed", error=err_to_str(err)
        )
    return workflow


# patching function as class method
kfp.compiler.Compiler._original_create_workflow = kfp.compiler.Compiler._create_workflow
kfp.compiler.Compiler._create_workflow = _create_enriched_mlrun_workflow
