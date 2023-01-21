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
import re
import subprocess
import tempfile

import click
import requests

import mlrun.utils

logger = mlrun.utils.create_logger(level="debug", name="automation")


class ReleaseNotesGenerator:
    commit_regex = (
        r"^"
        r"(?P<commitId>[a-zA-Z0-9]+)"
        r" {"
        r"(?P<username>[a-zA-Z0-9-_\s]+)"
        r"} "
        r"(\[(?P<scope>[^\]]*)\])?"
        r"( )?"
        r"(?P<commitMessage>.*?)"
        r"[ ]*"
        r"\("
        r"(?P<pullRequestNumber>#[0-9]+)"
        r"\)"
        r"$"
    )
    bug_fixes_words = ["fix", "bug"]

    def __init__(
        self,
        release: str,
        previous_release: str,
        release_branch: str,
        raise_on_failed_parsing: bool = True,
        tmp_file_path: str = None,
        skip_clone: bool = False,
    ):
        self._logger = logger
        self._release = release
        self._previous_release = previous_release
        self._release_branch = release_branch
        self._raise_on_failed_parsing = raise_on_failed_parsing
        self._tmp_file_path = tmp_file_path
        self._skip_clone = skip_clone
        # adding a map with the common contributors to prevent going to github API on every commit (better performance,
        # and prevent rate limiting)
        self._git_to_github_usernames_map = {
            "Hedingber": "Hedingber",
            "gilad-shaham": "gilad-shaham",
            "Saar Cohen": "theSaarco",
            "Yaron Haviv": "yaronha",
            "Liran BG": "liranbg",
            "Gal Topper": "gtopper",
            "guy1992l": "guy1992l",
            "Nick Brown": "ihs-nick",
            "Tom Tankilevitch": "tankilevitch",
            "Adam": "quaark",
            "Alon Maor": "AlonMaor14",
            "TomerShor": "TomerShor",
            "Assaf Ben-Amitai": "assaf758",
            "Yael Genish": "yaelgen",
        }

    def run(self):
        self._logger.info(
            "Generating release notes",
            release=self._release,
            previous_release=self._previous_release,
            release_branch=self._release_branch,
        )

        with tempfile.TemporaryDirectory(
            suffix="mlrun-release-notes-clone"
        ) as repo_dir:
            current_working_dir = repo_dir
            if self._skip_clone:
                current_working_dir = None
                self._logger.info(
                    "Skipping cloning repo, assuming already cloned, using current working dir"
                )
            else:
                self._logger.info("Cloning repo", repo_dir=current_working_dir)
                self._run_command(
                    "git",
                    args=[
                        "clone",
                        "--branch",
                        self._release_branch,
                        "git@github.com:mlrun/mlrun.git",
                        current_working_dir,
                    ],
                )

            commits_for_highlights = self._run_command(
                "git",
                args=[
                    "log",
                    '--pretty=format:"%h {%an} %s"',
                    f"{self._previous_release}..{self._release}",
                ],
                cwd=current_working_dir,
            )

            commits_for_pull_requests = self._run_command(
                "git",
                args=[
                    "log",
                    '--pretty=format:"%h %s"',
                    f"{self._previous_release}..{self._release}",
                ],
                cwd=current_working_dir,
            )

        self._generate_release_notes_from_commits(
            commits_for_highlights, commits_for_pull_requests
        )

    def _generate_release_notes_from_commits(
        self, commits_for_highlights, commits_for_pull_requests
    ):
        (
            feature_notes,
            bug_fixes_notes,
            failed_parsing_commits,
        ) = self._generate_highlight_notes_from_commits(commits_for_highlights)
        # TODO: enforce a commit message convention which will allow to parse whether it's a feature/enhancement or
        #  bug fix
        failed_commits = "\n".join(failed_parsing_commits)
        ui_feature_notes = (
            f"* **UI**: [Features & enhancement](https://github.com/mlrun/ui/releases/tag/"
            f"{self._release}#features-and-enhancements)"
        )
        ui_bug_fix_notes = f"* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{self._release}#bug-fixes)"
        feature_notes += ui_feature_notes
        bug_fixes_notes += ui_bug_fix_notes
        release_notes = f"""
### Features / Enhancements
{feature_notes}

### Bug fixes
{bug_fixes_notes}


#### Pull requests:
{commits_for_pull_requests}
"""

        if failed_parsing_commits:
            failed_parsing_template = f"""
#### Failed parsing:
{failed_commits}
"""
            release_notes += failed_parsing_template
            if self._raise_on_failed_parsing:
                self.output_release_notes(release_notes)
                raise ValueError(
                    "Failed parsing some of the commits, added them at the end of the release notes"
                )

        self.output_release_notes(release_notes)

    def output_release_notes(self, release_notes: str):
        print(release_notes)
        if self._tmp_file_path:
            logger.info("Writing release notes to file", path=self._tmp_file_path)
            with open(self._tmp_file_path, "w") as f:
                f.write(release_notes)

    def _generate_highlight_notes_from_commits(self, commits: str):
        feature_notes = ""
        bug_fixes_notes = ""
        failed_parsing_commits = []
        if commits:
            for commit in commits.split("\n"):
                match = re.fullmatch(self.commit_regex, commit)
                if match:
                    scope = match.groupdict()["scope"] or "Unknown"
                    message = match.groupdict()["commitMessage"]
                    pull_request_number = match.groupdict()["pullRequestNumber"]
                    commit_id = match.groupdict()["commitId"]
                    username = match.groupdict()["username"]
                    github_username = self._resolve_github_username(commit_id, username)
                    message_note = f"* **{scope}**: {message}, {pull_request_number}, @{github_username}\n"
                    if self._is_bug_fix(message_note):
                        bug_fixes_notes += message_note
                    else:
                        feature_notes += message_note
                elif commit:
                    failed_parsing_commits.append(commit)

        return feature_notes, bug_fixes_notes, failed_parsing_commits

    def _is_bug_fix(self, note: str):
        for bug_fix_word in self.bug_fixes_words:
            if bug_fix_word in note.lower():
                return True

        return False

    def _resolve_github_username(self, commit_id, username):
        """
        The username we get here is coming from the output of git log command, so it's the git username.
        We want to resolve the Github username, since when using these, Github automatically make the name clickable
        (links to the Github profile)
        Note that if for some reason we couldn't find the Github username we default to the git username which won't be
        clickable but at least will give the author name
        To prevent getting rate limit from Github API a static map (git username -> github username) of common
        contributors was added
        """
        if username in self._git_to_github_usernames_map:
            return self._git_to_github_usernames_map[username]
        response = requests.get(
            f"https://api.github.com/repos/mlrun/mlrun/commits/{commit_id}",
            # lock to v3 of the api to prevent breakages
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        default_username = username if username else "unknown"
        return response.json().get("author", {}).get("login", default_username)

    @staticmethod
    def _run_command(command, args=None, cwd=None):
        if args:
            command += " " + " ".join(args)

        try:
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                encoding="utf-8",
                cwd=cwd,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Command failed",
                stdout=exc.stdout,
                stderr=exc.stderr,
                return_code=exc.returncode,
                cmd=exc.cmd,
                args=exc.args,
            )
            raise
        output = process.stdout
        return output


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("release", type=str, required=True)
@click.argument("previous-release", type=str, required=True)
@click.argument("release-branch", type=str, required=False, default="master")
@click.argument("raise-on-failed-parsing", type=bool, required=False, default=True)
@click.argument("tmp-file-path", type=str, required=False, default=None)
@click.argument("skip-clone", type=bool, required=False, default=False)
def run(
    release: str,
    previous_release: str,
    release_branch: str,
    raise_on_failed_parsing: bool,
    tmp_file_path: str = None,
    skip_clone: bool = False,
):
    release_notes_generator = ReleaseNotesGenerator(
        release,
        previous_release,
        release_branch,
        raise_on_failed_parsing,
        tmp_file_path,
        skip_clone,
    )
    try:
        release_notes_generator.run()
    except Exception as exc:
        logger.error("Failed running release notes generator", exc=exc)
        raise


if __name__ == "__main__":
    main()
