import re

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
