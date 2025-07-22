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

ARG MLRUN_PYTHON_VERSION=3.11

FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

RUN apt-get update -qq && tar -C /var/lib/apt -cf /tmp/apt-lists.tar lists
RUN tar -C /var/lib/apt -xf /tmp/apt-lists.tar && rm /tmp/apt-lists.tar

# Update OS packages:
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -q