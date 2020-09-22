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
ARG MLRUN_PYTHON_VERSION=3.7

FROM python:${MLRUN_PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y \
  git-core

RUN python -m pip install --upgrade --no-cache pip

COPY ./automation/requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt

WORKDIR /mlrun

VOLUME /mlrun

# "sh" is needed to enforce running in a shell - otherwise env vars substitution won't happen
CMD ["sh", "-c", "python ./automation/version/version_file.py create $MLRUN_VERSION"]
