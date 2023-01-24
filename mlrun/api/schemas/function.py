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
#
import typing

import pydantic

import mlrun.api.utils.helpers


# Ideally we would want this to be class FunctionState(mlrun.api.utils.helpers.StrEnum) which is the
# "FastAPI-compatible" way of creating schemas
# But, when we save a function to the DB, we pickle the body, which saves the state as an instance of this class (and
# not just a string), then if for some reason we downgrade to 0.6.4, before we had this class, we fail reading (pickle
# load) the function from the DB.
# Note that the problems are happening only if the state is assigned in the API side. When it's in the client side it
# anyways passes through JSON in the HTTP request body and come up as a string in the API side.
# For now I'm simply making the class a simple string consts class
# 2 other solutions I thought of:
# 1. Changing the places where we set the state in the UI to use set the actual enum value (FunctionState.x.value) - too
# fragile, tomorrow someone will set the state using the enum
# 2. Changing the function to be saved into a JSON field instead of pickled inside BLOB field - looks like the ideal we
# should go to, but too complicated and needed something fast and simple.
class FunctionState:
    unknown = "unknown"
    ready = "ready"
    error = "error"  # represents deployment error

    deploying = "deploying"
    # there is currently an abuse usage of the builder (lower) pod state as the function state, ideally these two would
    # map to deploying but for backwards compatibility reasons we have to keep them
    running = "running"
    pending = "pending"
    # same goes for the build which is not coming from the pod, but is used and we can't just omit it for BC reasons
    build = "build"


class PreemptionModes(mlrun.api.utils.helpers.StrEnum):
    # makes function pods be able to run on preemptible nodes
    allow = "allow"
    # makes the function pods run on preemptible nodes only
    constrain = "constrain"
    # prevents the function pods from running on preemptible nodes
    prevent = "prevent"
    # doesn't apply any preemptible node selection on the function
    none = "none"


# used when running in Iguazio (otherwise use disabled mode)
# populates mlrun.mlconf.function.spec.security_context.enrichment_mode
class SecurityContextEnrichmentModes(mlrun.api.utils.helpers.StrEnum):
    # always use the user id of the user that triggered the 1st run / created the function
    # NOTE: this mode is incomplete and not fully supported yet
    retain = "retain"
    # use the user id of the user that triggered the current run
    override = "override"
    # security context is not auto applied
    disabled = "disabled"


class ImagePullSecret(pydantic.BaseModel):
    default: typing.Optional[str]


class Pipelines(pydantic.BaseModel):
    kfp_pod_user_unix_id: typing.Optional[int]


class SecurityContext(pydantic.BaseModel):
    default: typing.Optional[str]
    enrichment_mode: typing.Optional[SecurityContextEnrichmentModes]
    enrichment_group_id: typing.Optional[int]
    pipelines: typing.Optional[Pipelines]


class ServiceAccount(pydantic.BaseModel):
    default: typing.Optional[str]


class FunctionSpec(pydantic.BaseModel):
    image_pull_secret: typing.Optional[ImagePullSecret]
    security_context: typing.Optional[SecurityContext]
    service_account: typing.Optional[ServiceAccount]


class Function(pydantic.BaseModel):
    spec: typing.Optional[FunctionSpec]
