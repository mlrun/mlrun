import unittest.mock

import pytest
import mlrun.utils.clones


@pytest.mark.parametrize(
    "ref,ref_type",
    [
        ("without-slash", "branch"),
        ("with/slash", "branch"),
        ("without-slash", "tag"),
        ("without/slash", "tag"),
    ],
)
def test_clone_git_refs(ref, ref_type):
    repo = "github.com/some-git-project/some-git-repo.git"
    url = f"git://{repo}#refs/{'heads' if ref_type == 'branch' else 'tags'}/{ref}"
    context = "non-existent-dir"
    branch = ref if ref_type == "branch" else None
    tag = ref if ref_type == "tag" else None

    with unittest.mock.patch("git.Repo.clone_from") as clone_from:
        _, repo_obj = mlrun.utils.clones.clone_git(url, context)
        clone_from.assert_called_once_with(
            f"https://{repo}", context, single_branch=True, b=branch
        )
        if tag:
            repo_obj.git.checkout.assert_called_once_with(tag)
