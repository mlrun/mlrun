import re
import automation.release_notes.generate


def test_commit_regex_matching():
    cases = [
        {
            "commit_line": "595a69d6 [MPIJob] Support setting mpi args [Backport 0.5.x] (#705)",
            "expected_commit_id": "595a69d6",
            "expected_scope": "MPIJob",
            "expected_commit_message": "Support setting mpi args [Backport 0.5.x]",
            "expected_pull_request_number": "#705",
        },
        {
            "commit_line": "a654a04e [CI] Fixing release branch automation to include file deletions (#685)",
            "expected_commit_id": "a654a04e",
            "expected_scope": "CI",
            "expected_commit_message": "Fixing release branch automation to include file deletions",
            "expected_pull_request_number": "#685",
        },
        {
            "commit_line": "ced0e66 [Builder] Make init container image configurable, set pull policy and pin version ("
            "3.13.1) (#696)",
            "expected_commit_id": "ced0e66",
            "expected_scope": "Builder",
            "expected_commit_message": "Make init container image configurable, set pull policy and pin version (3.13.1"
            ")",
            "expected_pull_request_number": "#696",
        },
        {
            "commit_line": "8f0d394 few small doc edits in dask, how-to, and tutorial (#692)",
            "expected_commit_id": "8f0d394",
            "expected_scope": None,
            "expected_commit_message": "few small doc edits in dask, how-to, and tutorial",
            "expected_pull_request_number": "#692",
        },
    ]

    for case in cases:
        match = re.fullmatch(
            automation.release_notes.generate.ReleaseNotesGenerator.commit_regex,
            case["commit_line"],
        )
        assert match is not None, f"Commit did not matched regex. {case['commit_line']}"
        assert case["expected_commit_id"] == match.groupdict()["commitId"]
        assert case["expected_scope"] == match.groupdict()["scope"]
        assert case["expected_commit_message"] == match.groupdict()["commitMessage"]
        assert (
            case["expected_pull_request_number"]
            == match.groupdict()["pullRequestNumber"]
        )
