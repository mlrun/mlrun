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

    def __init__(
        self,
        release: str,
        previous_release: str,
        release_branch: str,
    ):
        self._logger = logger
        self._release = release
        self._previous_release = previous_release
        self._release_branch = release_branch
        # adding a map with the common contributors to prevent going to github API on every commit (better performance,
        # and prevent rate limiting)
        self._git_to_github_usernames_map = {
            "Hedingber": "Hedingber",
            "gilad-shaham": "gilad-shaham",
            "Saar Cohen": "theSaarco",
            "urihoenig": "urihoenig",
            "Eyal Salomon": "eyalsol",
            "Katya Katsenelenbogen": "katyakats",
            "Ben": "benbd86",
            "Yaron Haviv": "yaronha",
            "Marcelo Litovsky": "marcelonyc",
            "Liran BG": "liranbg",
            "Gal Topper": "gtopper",
            "Dina Nimrodi": "dinal",
            "Michael": "Michaelliv",
            "guy1992l": "guy1992l",
            "Nick Brown": "ihs-nick",
            "Oded Messer": "omesser",
            "Tom Tankilevitch": "tankilevitch",
            "Adam": "quaark",
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
            self._logger.info("Cloning repo", repo_dir=repo_dir)
            self._run_command(
                "git",
                args=[
                    "clone",
                    "--branch",
                    self._release_branch,
                    "git@github.com:mlrun/mlrun.git",
                    repo_dir,
                ],
            )

            commits_for_highlights = self._run_command(
                "git",
                args=[
                    "log",
                    '--pretty=format:"%h {%an} %s"',
                    f"{self._previous_release}..{self._release}",
                ],
                cwd=repo_dir,
            )

            commits_for_pull_requests = self._run_command(
                "git",
                args=[
                    "log",
                    '--pretty=format:"%h %s"',
                    f"{self._previous_release}..{self._release}",
                ],
                cwd=repo_dir,
            )

        self._generate_release_notes_from_commits(
            commits_for_highlights, commits_for_pull_requests
        )

    def _generate_release_notes_from_commits(
        self, commits_for_highlights, commits_for_pull_requests
    ):
        (
            highlight_notes,
            failed_parsing_commits,
        ) = self._generate_highlight_notes_from_commits(commits_for_highlights)
        # currently we just put everything under features / enhancements
        # TODO: enforce a commit message convention which will allow to parse whether it's a feature/enhancement or
        #  bug fix
        failed_commits = "\n".join(failed_parsing_commits)
        release_notes = f"""
### Features / Enhancements
{highlight_notes}
* **UI**: [Features & enhancement](https://github.com/mlrun/ui/releases/tag/{self._release}#features-and-enhancements)

### Bug fixes
* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{self._release}#bug-fixes)


#### Pull requests:
{commits_for_pull_requests}
"""

        if failed_parsing_commits:
            failed_parsing_template = f"""
#### Failed parsing:
{failed_commits}
"""

            print(release_notes + failed_parsing_template)
            raise ValueError(
                "Failed parsing some of the commits, added them at the end of the release notes"
            )

        print(release_notes)

    def _generate_highlight_notes_from_commits(self, commits: str):
        highlighted_notes = ""
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
                    highlighted_notes += f"* **{scope}**: {message}, {pull_request_number}, @{github_username}\n"
                elif commit:
                    failed_parsing_commits.append(commit)

        return highlighted_notes, failed_parsing_commits

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
def run(
    release: str,
    previous_release: str,
    release_branch: str,
):
    release_notes_generator = ReleaseNotesGenerator(
        release, previous_release, release_branch
    )
    try:
        release_notes_generator.run()
    except Exception as exc:
        logger.error("Failed running release notes generator", exc=exc)
        raise


if __name__ == "__main__":
    main()
