import re
import subprocess
import tempfile

import click

import mlrun.utils

logger = mlrun.utils.create_logger(level="debug", name="automation")


class ReleaseNotesGenerator:
    def __init__(
        self, release: str, previous_release: str, release_branch: str,
    ):
        self._logger = logger
        self._release = release
        self._previous_release = previous_release
        self._release_branch = release_branch

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

            commits = self._run_command(
                "git",
                args=[
                    "log",
                    '--pretty=format:"%h %s"',
                    f"{self._previous_release}..HEAD",
                ],
                cwd=repo_dir,
            )

        self._generate_release_notes_from_commits(commits)

    def _generate_release_notes_from_commits(self, commits):
        highlight_notes = self._generate_highlight_notes_from_commits(commits)
        # currently we just put everything under features / enhancements
        # TODO: enforce a commit message convention which will allow to parse whether it's a feature/enhancement or
        #  bug fix
        print(
            f"""
### Features / Enhancements
{highlight_notes}
* **UI**: [Features & enhancment](https://github.com/mlrun/ui/releases/tag/{self._release}#features-and-enhancements)

### Bug fixes
* **UI**: [Bug fixes](https://github.com/mlrun/ui/releases/tag/{self._release}#bug-fixes)


#### Pull requests:
{commits}
        """
        )

    def _generate_highlight_notes_from_commits(self, commits):
        regex = (
            r"^"
            r"(?P<commitId>[a-zA-Z0-9]+)"
            r" "
            r"(\[(?P<scope>.*)\])?"
            r"(?P<commitMessage>.*)"
            r"\("
            r"(?P<pullRequestNumber>#[0-9]+)"
            r"\)"
            r"$"
        )
        highlighted_notes = ""
        for commit in commits.split("\n"):
            match = re.fullmatch(regex, commit)
            assert match is not None, f"Commit did not matched regex. {commit}"
            scope = match.groupdict()["scope"] or "Unknown"
            message = match.groupdict()["commitMessage"]
            pull_request_number = match.groupdict()["pullRequestNumber"]
            # currently just defaulting to hedingber TODO: resolve the real author name
            highlighted_notes += (
                f"* **{scope}**: {message}, {pull_request_number}, @hedingber\n"
            )

        return highlighted_notes

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
    release: str, previous_release: str, release_branch: str,
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
