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
import io
import re
import unittest.mock

import deepdiff

import automation.release_notes.generate


def test_commit_regex_matching():
    cases = [
        {
            "commit_line": "595a69d6 {Or Zilberman} [MPIJob] Support setting mpi args [Backport 0.5.x] (#705)",
            "expected_commit_id": "595a69d6",
            "expected_username": "Or Zilberman",
            "expected_scope": "MPIJob",
            "expected_commit_message": "Support setting mpi args [Backport 0.5.x]",
            "expected_pull_request_number": "#705",
        },
        {
            "commit_line": "a654a04e {Hedingber} [CI] Fixing release branch automation to include file deletions (#685)"
            "",
            "expected_commit_id": "a654a04e",
            "expected_username": "Hedingber",
            "expected_scope": "CI",
            "expected_commit_message": "Fixing release branch automation to include file deletions",
            "expected_pull_request_number": "#685",
        },
        {
            "commit_line": "ced0e66 {urihoenig} [Builder] Make init container image configurable, set pull policy and p"
            "in version (3.13.1) (#696)",
            "expected_commit_id": "ced0e66",
            "expected_username": "urihoenig",
            "expected_scope": "Builder",
            "expected_commit_message": "Make init container image configurable, set pull policy and pin version (3.13.1"
            ")",
            "expected_pull_request_number": "#696",
        },
        {
            "commit_line": "8f0d394 {Yaron Haviv} few small doc edits in dask, how-to, and tutorial (#692)",
            "expected_commit_id": "8f0d394",
            "expected_username": "Yaron Haviv",
            "expected_scope": None,
            "expected_commit_message": "few small doc edits in dask, how-to, and tutorial",
            "expected_pull_request_number": "#692",
        },
        {
            "commit_line": "1540304f {gilad-shaham} Update howto converting to MLRun to use set_environment (#689)",
            "expected_commit_id": "1540304f",
            "expected_username": "gilad-shaham",
            "expected_scope": None,
            "expected_commit_message": "Update howto converting to MLRun to use set_environment",
            "expected_pull_request_number": "#689",
        },
        {
            "commit_line": "45e91253 {guy1992l} [Frameworks] Added H5 saving format (#1055)",
            "expected_commit_id": "45e91253",
            "expected_username": "guy1992l",
            "expected_scope": "Frameworks",
            "expected_commit_message": "Added H5 saving format",
            "expected_pull_request_number": "#1055",
        },
        {
            "commit_line": "a654a04e {Hedingber} [CI] Mistakenly removing the space(#685)"
            "",
            "expected_commit_id": "a654a04e",
            "expected_username": "Hedingber",
            "expected_scope": "CI",
            "expected_commit_message": "Mistakenly removing the space",
            "expected_pull_request_number": "#685",
        },
    ]

    for case in cases:
        match = re.fullmatch(
            automation.release_notes.generate.ReleaseNotesGenerator.commit_regex,
            case["commit_line"],
        )
        assert match is not None, f"Commit did not matched regex. {case['commit_line']}"
        assert case["expected_commit_id"] == match.groupdict()["commitId"]
        assert case["expected_username"] == match.groupdict()["username"]
        assert case["expected_scope"] == match.groupdict()["scope"]
        assert case["expected_commit_message"] == match.groupdict()["commitMessage"]
        assert (
            case["expected_pull_request_number"]
            == match.groupdict()["pullRequestNumber"]
        )


def test_generate_release_notes():
    release = "v0.9.0-rc8"
    previous_release = "v0.9.0-rc7"
    release_branch = "master"

    release_generator = automation.release_notes.generate.ReleaseNotesGenerator(
        release, previous_release, release_branch
    )

    cases = [
        {
            "_run_command": [
                None,
                "fd6c5a86 {Gal Topper} [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)\n"
                "985d7cb8 {Saar Cohen} [Secrets] Verify project secrets do not exist when deleting a project (#1552)",
                "fd6c5a86 [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)\n"
                "985d7cb8 [Secrets] Verify project secrets do not exist when deleting a project (#1552)",
            ],
            "_resolve_github_username": ["gtopper", "theSaarco"],
            "expected_response": f"""
### Features / Enhancements
* **Requirements**: Bump storey to 0.8.15 and v3io-frames to 0.10.2, #1553, @gtopper
* **Secrets**: Verify project secrets do not exist when deleting a project, #1552, @theSaarco

* **UI**: [Features & enhancement](https://github.com/mlrun/ui/releases/tag/{release}#features-and-enhancements)

### Bug fixes
* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{release}#bug-fixes)


#### Pull requests:
fd6c5a86 [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)
985d7cb8 [Secrets] Verify project secrets do not exist when deleting a project (#1552)

""",
        },
        {
            "_run_command": [
                None,
                "fd6c5a86 {Gal Topper} [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)\n"
                "20d4088c {yuribros1974} Merge pull request #1511 from mlrun/ML-509_update_release_status\n"
                "985d7cb8 {Saar Cohen} [Secrets] Verify project secrets do not exist when deleting a project (#1552)",
                "fd6c5a86 [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)\n"
                "20d4088c Merge pull request #1511 from mlrun/ML-509_update_release_status\n"
                "985d7cb8 [Secrets] Verify project secrets do not exist when deleting a project (#1552)",
            ],
            "_resolve_github_username": ["gtopper", "theSaarco"],
            "expect_failure": True,
            "expected_response": f"""
### Features / Enhancements
* **Requirements**: Bump storey to 0.8.15 and v3io-frames to 0.10.2, #1553, @gtopper
* **Secrets**: Verify project secrets do not exist when deleting a project, #1552, @theSaarco

* **UI**: [Features & enhancement](https://github.com/mlrun/ui/releases/tag/{release}#features-and-enhancements)

### Bug fixes
* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{release}#bug-fixes)


#### Pull requests:
fd6c5a86 [Requirements] Bump storey to 0.8.15 and v3io-frames to 0.10.2 (#1553)
20d4088c Merge pull request #1511 from mlrun/ML-509_update_release_status
985d7cb8 [Secrets] Verify project secrets do not exist when deleting a project (#1552)

#### Failed parsing:
20d4088c {{yuribros1974}} Merge pull request #1511 from mlrun/ML-509_update_release_status

""",
        },
        {
            "_run_command": [None, "", ""],
            "_resolve_github_username": None,
            "expect_failure": False,
            "expected_response": f"""
### Features / Enhancements

* **UI**: [Features & enhancement](https://github.com/mlrun/ui/releases/tag/{release}#features-and-enhancements)

### Bug fixes
* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{release}#bug-fixes)


#### Pull requests:


""",
        },
    ]
    automation.release_notes.generate.tempfile = unittest.mock.MagicMock()
    for case in cases:
        with unittest.mock.patch(
            "automation.release_notes.generate.ReleaseNotesGenerator._run_command"
        ) as _run_command_mock, unittest.mock.patch(
            "automation.release_notes.generate.ReleaseNotesGenerator._resolve_github_username"
        ) as _resolve_github_user_mock, unittest.mock.patch(
            "sys.stdout", new=io.StringIO()
        ) as stdout_mock:
            _run_command_mock.side_effect = case["_run_command"]
            _resolve_github_user_mock.side_effect = case["_resolve_github_username"]
            try:
                release_generator.run()
            except ValueError:
                if not case.get("expect_failure", False):
                    raise
            diff = deepdiff.DeepDiff(case["expected_response"], stdout_mock.getvalue())
            assert diff == {}
