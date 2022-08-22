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
import json
import re
import subprocess

import click

import mlrun.utils

logger = mlrun.utils.create_logger(level="debug", name="automation")


class PackageTester:
    def __init__(self):
        self._logger = logger

        basic_import = "import mlrun"
        api_import = "import mlrun.api.main"
        s3_import = "import mlrun.datastore.s3"
        azure_blob_storage_import = "import mlrun.datastore.azure_blob"
        azure_key_vault_import = "import mlrun.utils.azure_vault"
        google_cloud_bigquery_import = (
            "from mlrun.datastore.sources import BigQuerySource"
        )
        google_cloud_storage_import = "import mlrun.datastore.google_cloud_storage"
        targets_import = "import mlrun.datastore.targets"
        redis_import = "import redis"

        self._extras_tests_data = {
            "": {"import_test_command": f"{basic_import}"},
            "[api]": {"import_test_command": f"{basic_import}; {api_import}"},
            "[complete-api]": {
                "import_test_command": f"{basic_import}; {api_import}; {s3_import}; "
                + f"{azure_blob_storage_import}; {azure_key_vault_import}",
                "perform_vulnerability_check": True,
            },
            "[s3]": {"import_test_command": f"{basic_import}; {s3_import}"},
            "[azure-blob-storage]": {
                "import_test_command": f"{basic_import}; {azure_blob_storage_import}"
            },
            "[azure-key-vault]": {
                "import_test_command": f"{basic_import}; {azure_key_vault_import}"
            },
            # TODO: this won't actually fail if the requirement is missing
            "[google-cloud-bigquery]": {
                "import_test_command": f"{basic_import}; {google_cloud_bigquery_import}"
            },
            "[google-cloud-storage]": {
                "import_test_command": f"{basic_import}; {google_cloud_storage_import}"
            },
            # TODO: this won't actually fail if the requirement is missing
            "[kafka]": {"import_test_command": f"{basic_import}; {targets_import}"},
            "[complete]": {
                "import_test_command": f"{basic_import}; {s3_import}; {azure_blob_storage_import}; "
                + f"{azure_key_vault_import}; {google_cloud_storage_import}; {targets_import}",
                "perform_vulnerability_check": True,
            },
            "[redis]": {
                "import_test_command": f"{basic_import}; {redis_import}"
            },
        }

    def run(self):
        self._logger.info(
            "Running package tests",
        )

        results = {}
        for extra, extra_tests_data in self._extras_tests_data.items():
            self._create_venv()
            self._install_extra(extra)
            results[extra] = {}
            self._run_test(self._test_extra_imports, extra, results, "import_test")
            self._run_test(
                self._test_requirements_conflicts,
                extra,
                results,
                "requirements_conflicts_test",
            )
            if extra_tests_data.get("perform_vulnerability_check"):
                self._run_test(
                    self._test_requirements_vulnerabilities,
                    extra,
                    results,
                    "requirements_vulnerabilities_test",
                )
            self._clean_venv()

        failed = False
        for extra_tests_results in results.values():
            if (
                not extra_tests_results["import_test"]["passed"]
                or not extra_tests_results["requirements_conflicts_test"]["passed"]
            ):
                failed = True
                break

        self._logger.info(
            "Finished running package tests", results=results, failed=failed
        )
        if failed:
            raise RuntimeError("Package tests failed")

    def _run_test(self, test_function, extra, results, test_key):
        try:
            test_function(extra)
        except Exception:
            results[extra].setdefault(test_key, {})["passed"] = False
        else:
            results[extra].setdefault(test_key, {})["passed"] = True

    def _test_extra_imports(self, extra):
        self._logger.debug(
            "Testing extra imports",
            extra=extra,
        )
        test_command = (
            f"python -c '{self._extras_tests_data[extra]['import_test_command']}'"
        )
        self._run_command(
            test_command,
            run_in_venv=True,
        )
        if "api" not in extra:
            # When api is not in the extra it's an extra purposed for the client usage
            # Usually it will be used with a remote server - meaning httpdb as the run db, to test that (will cause
            # different imports to be done) we're setting the DB path
            self._run_command(
                test_command,
                run_in_venv=True,
                env={"MLRUN_DBPATH": "http://mock-server"},
            )

    def _test_requirements_vulnerabilities(self, extra):
        """
        When additional vulnerabilities are being discovered, we expect this test to fail; our objective is to fix them
        all ASAP. There are several circumstances when this cannot be handled simply by using a newer version, such as
        when another library requires an older version, a vulnerability that has not been fixed, or an environment
        requirement. If this is the case, we will add the library with a thorough description of the issue and suggested
        action items for future readers to the ignored_vulnerabilities.
        When a vulnerability is fixed, we should remove it from the ignored_vulnerabilities list.
        """
        self._logger.debug(
            "Testing requirements vulnerabilities",
            extra=extra,
        )
        self._run_command(
            "pip install safety",
            run_in_venv=True,
        )
        code, stdout, stderr = self._run_command(
            "safety check --json",
            run_in_venv=True,
            raise_on_error=False,
        )
        if code != 0:
            vulnerabilities = json.loads(stdout)
            if vulnerabilities:
                self._logger.debug(
                    "Found requirements vulnerabilities",
                    vulnerabilities=vulnerabilities,
                )
            ignored_vulnerabilities = {
                "kubernetes": [
                    {
                        "pattern": r"^Kubernetes(.*)unfixed vulnerability, CVE-2021-29923(.*)",
                        "reason": "Vulnerability not fixed, nothing we can do",
                    }
                ],
                "mlrun": [
                    {
                        "pattern": r"^Mlrun(.*)TensorFlow' \(2.4.1\)(.*)$",
                        "reason": "Newer tensorflow versions are not compatible with our CUDA and rapids versions so we"
                        " can't upgrade it",
                    },
                    {
                        "pattern": r"(.*)"
                        r"("
                        r"https://github\.com/mlrun/mlrun/pull/1997/commits/de4c87f478f8d76dd8e46942588c81e"
                        r"f0d0b481e"
                        r"|"
                        r"1\.0\.3rc1 adds \"notebook~=6\.4"
                        r"|"
                        r"1\.0\.3rc1 adds \"pillow~=9\.0"
                        r")"
                        r"(.*)",
                        "reason": "Those already fixed, we're getting them only because in our CI our version is "
                        "0.0.0+unstable",
                    },
                ],
            }
            filtered_vulnerabilities = []
            for vulnerability in vulnerabilities:
                if vulnerability[0] in ignored_vulnerabilities:
                    ignored_vulnerability = ignored_vulnerabilities[vulnerability[0]]
                    ignore_vulnerability = False
                    for ignored_pattern in ignored_vulnerability:
                        if re.search(ignored_pattern["pattern"], vulnerability[3]):
                            self._logger.debug(
                                "Ignoring vulnerability",
                                vulnerability=vulnerability,
                                reason=ignored_pattern["reason"],
                            )
                            ignore_vulnerability = True
                            break
                    if ignore_vulnerability:
                        continue
                filtered_vulnerabilities.append(vulnerability)
            if filtered_vulnerabilities:
                message = "Found vulnerable requirements that can not be ignored"
                logger.warning(
                    message,
                    vulnerabilities=vulnerabilities,
                    filtered_vulnerabilities=filtered_vulnerabilities,
                    ignored_vulnerabilities=ignored_vulnerabilities,
                )
                raise AssertionError(message)

    def _test_requirements_conflicts(self, extra):
        self._logger.debug(
            "Testing requirements conflicts",
            extra=extra,
        )
        self._run_command(
            "pip install pipdeptree",
            run_in_venv=True,
        )
        self._run_command(
            "pipdeptree --warn fail",
            run_in_venv=True,
        )

    def _create_venv(self):
        self._logger.debug(
            "Creating venv",
        )
        self._run_command(
            "python -m venv test-venv",
        )

    def _clean_venv(self):
        self._logger.debug(
            "Cleaning venv",
        )
        self._run_command(
            "rm -rf test-venv",
        )

    def _install_extra(self, extra):
        self._logger.debug(
            "Installing extra",
            extra=extra,
        )
        self._run_command(
            "python -m pip install --upgrade pip~=22.0.0",
            run_in_venv=True,
        )

        self._run_command(
            f"pip install '.{extra}'",
            run_in_venv=True,
        )

    def _run_command(self, command, run_in_venv=False, env=None, raise_on_error=True):
        if run_in_venv:
            command = f". test-venv/bin/activate && {command}"
        try:
            process = subprocess.run(
                command,
                env=env,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            if raise_on_error:
                logger.warning(
                    "Command failed",
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    return_code=exc.returncode,
                    cmd=exc.cmd,
                    args=exc.args,
                )
                raise
            return exc.returncode, exc.stdout, exc.stderr
        return process.returncode, process.stdout, process.stderr


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
def run():
    package_tester = PackageTester()
    try:
        package_tester.run()
    except Exception as exc:
        logger.error("Failed running the package tester", exc=exc)
        raise


if __name__ == "__main__":
    main()
