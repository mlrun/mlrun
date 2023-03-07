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

import git

import mlrun.runtimes.utils


def test_add_code_metadata():
    # gets to the root of the git repo
    repo = git.Repo(search_parent_directories=True)
    code_metadata = mlrun.runtimes.utils.add_code_metadata(repo.git_dir)
    assert "mlrun.git" in code_metadata, "code metadata should contain git info"
    assert (
        repo.head.commit.hexsha in code_metadata
    ), "commit hash should be in code metadata"
