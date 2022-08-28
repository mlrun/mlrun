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
import mlrun.api.utils.projects.follower
import mlrun.api.utils.projects.leader
import mlrun.api.utils.projects.member
import mlrun.config

# TODO: something nicer
project_member: mlrun.api.utils.projects.member.Member = None


def initialize_project_member():
    global project_member
    if mlrun.config.config.httpdb.projects.leader in ["mlrun", "nop-self-leader"]:
        project_member = mlrun.api.utils.projects.leader.Member()
        project_member.initialize()
    else:
        project_member = mlrun.api.utils.projects.follower.Member()
        project_member.initialize()


def get_project_member() -> mlrun.api.utils.projects.member.Member:
    global project_member
    return project_member
