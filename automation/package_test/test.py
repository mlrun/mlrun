# Copyright 2023 Iguazio
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

import json
import os
import re
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import click

import mlrun.utils


def _test_extra_worker(logger, extra, extra_tests_data, package_installer):
    """
    Worker function for parallel execution.
    This is a module-level function so it can be pickled for ProcessPoolExecutor.
    """
    pid = os.getpid()
    worker_logger = logger.get_child(f"pid-{pid}.{extra or 'base'}")
    normalized_extra = extra.replace("[", "").replace("]", "")
    venv_name = f"extra-{normalized_extra}"
    venv_name = f"test-venvs/{venv_name}".rstrip("-")

    result = {"extra": extra, "tests": {}}

    try:
        with Venv(worker_logger, package_installer, venv_name) as venv:
            venv.install_extra(extra)

            # Test imports
            try:
                _run_import_test(venv, extra, extra_tests_data, worker_logger)
                result["tests"]["import_test"] = {"passed": True}
            except Exception as e:
                worker_logger.warning(f"Import test failed for {extra}", error=str(e))
                result["tests"]["import_test"] = {"passed": False}

            # Test requirements conflicts
            try:
                _run_conflicts_test(venv, extra, worker_logger)
                result["tests"]["requirements_conflicts_test"] = {"passed": True}
            except Exception as e:
                worker_logger.warning(
                    f"Conflicts test failed for {extra}", error=str(e)
                )
                result["tests"]["requirements_conflicts_test"] = {"passed": False}

            # Test vulnerabilities (if needed)
            if extra_tests_data.get("perform_vulnerability_check"):
                try:
                    _run_vulnerability_test(venv, extra, worker_logger)
                    result["tests"]["requirements_vulnerabilities_test"] = {
                        "passed": True
                    }
                except Exception as e:
                    worker_logger.warning(
                        f"Vulnerability test failed for {extra}", error=str(e)
                    )
                    result["tests"]["requirements_vulnerabilities_test"] = {
                        "passed": False
                    }
    except Exception as e:
        worker_logger.error(f"Failed to test extra {extra}", error=str(e))
        result["tests"]["import_test"] = {"passed": False}
        result["tests"]["requirements_conflicts_test"] = {"passed": False}

    return result


def _run_import_test(venv, extra, extra_tests_data, worker_logger):
    worker_logger.debug("Testing extra imports", extra=extra)
    env = {"MLRUN_LOG_LEVEL": "DEBUG"}
    test_command = f"python -c '{extra_tests_data['import_test_command']}'"
    venv.run_command(test_command, env=env)
    if "api" not in extra:
        venv.run_command(test_command, env=env | {"MLRUN_DBPATH": "http://mock-server"})


def _run_conflicts_test(venv, extra, worker_logger):
    worker_logger.debug("Testing requirements conflicts", extra=extra)
    venv.install_package("pipdeptree<2.29.0")
    venv.run_command("pipdeptree --warn fail")


def _run_vulnerability_test(venv, extra, worker_logger):
    worker_logger.debug("Testing requirements vulnerabilities", extra=extra)
    venv.install_package("safety")
    code, stdout, _ = venv.run_command("safety check --json", raise_on_error=False)
    if code != 0:
        full_report = json.loads(stdout)
        vulnerabilities = full_report["vulnerabilities"]
        if vulnerabilities:
            worker_logger.debug(
                "Found requirements vulnerabilities", vulnerabilities=vulnerabilities
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
                    "reason": "Newer tensorflow versions are not compatible with our CUDA and rapids versions",
                },
                {
                    "pattern": (
                        r"(.*)(https://github\.com/mlrun/mlrun/pull/1997/commits/"
                        r"de4c87f478f8d76dd8e46942588c81ef0d0b481e|1\.0\.3rc1 adds "
                        r"\"notebook~=6\.4|1\.0\.3rc1 adds \"pillow~=9\.0)(.*)"
                    ),
                    "reason": "Already fixed, we're getting them only because in CI our version is 0.0.0+unstable",
                },
            ],
        }

        filtered_vulnerabilities = []
        for vulnerability in vulnerabilities:
            if vulnerability["package_name"] in ignored_vulnerabilities:
                ignored_vulnerability = ignored_vulnerabilities[
                    vulnerability["package_name"]
                ]
                ignore_vulnerability = False
                for ignored_pattern in ignored_vulnerability:
                    if re.search(ignored_pattern["pattern"], vulnerability["advisory"]):
                        worker_logger.debug(
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
            worker_logger.warning(
                message,
                filtered_vulnerabilities=filtered_vulnerabilities,
                ignored_vulnerabilities=ignored_vulnerabilities,
            )
            raise AssertionError(message)


class Venv:
    def __init__(self, logger, package_manager="pip", venv_name="test-venv"):
        self._venv_name = venv_name
        self._package_manager = package_manager
        self._package_installer_command = (
            "uv pip install"
            if self._package_manager == "uv"
            else "python -m pip install"
        )
        self._logger = logger

    def __enter__(self):
        self._create_venv()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean_venv()

    def install_extra(self, extra):
        self._logger.debug(
            "Installing extra",
            extra=extra,
        )
        self.install_package("pip~=25.0.0", upgrade=True)
        self.install_package(f".{extra}")

    def install_package(self, package, upgrade=False):
        cmd = self._package_installer_command
        if upgrade:
            cmd += " --upgrade"
        cmd += f" {shlex.quote(package)}"
        return self.run_command(cmd)

    def run_command(self, command, env=None, raise_on_error=True):
        venv_command = f". {self._venv_name}/bin/activate"
        command = f"{venv_command} && {command}"
        return self._run_command(command, env=env, raise_on_error=raise_on_error)

    def _create_venv(self):
        self._logger.debug(
            "Creating venv",
            venv_name=self._venv_name,
        )
        self._run_command(f"python -m venv {self._venv_name}")

    def _clean_venv(self):
        self._logger.debug(
            "Cleaning venv",
            venv_name=self._venv_name,
        )
        self._run_command(f"rm -rf {self._venv_name}")

    def _run_command(
        self,
        command,
        env=None,
        raise_on_error=True,
    ):
        process = subprocess.Popen(
            command,
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for streaming
            text=True,
        )
        output_lines = []
        for line in process.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            self._logger.debug(f"> {line}")
        process.wait()
        output = "\n".join(output_lines)
        if process.returncode != 0 and raise_on_error:
            self._logger.warning(
                "Command failed",
                output=output,
                return_code=process.returncode,
                cmd=command,
            )
            raise subprocess.CalledProcessError(process.returncode, command, output, "")
        return process.returncode, output, ""


class PackageTester:
    def __init__(self, logger):
        self._logger = logger.get_child("package_tester")
        self._package_installer = os.getenv("MLRUN_PYTHON_PACKAGE_INSTALLER", "pip")

        basic_import = "import mlrun"
        s3_import = "import mlrun.datastore.s3"
        azure_blob_storage_import = "import mlrun.datastore.azure_blob"
        azure_key_vault_import = "import mlrun.utils.azure_vault"
        google_cloud_bigquery_import = (
            "from mlrun.datastore.sources import BigQuerySource"
        )
        oss_import = "import mlrun.datastore.alibaba_oss"
        google_cloud_storage_import = "import mlrun.datastore.google_cloud_storage"
        targets_import = "import mlrun.datastore.targets"
        redis_import = "import redis"
        mlflow_import = "import mlflow"

        self._extras_tests_data = {
            "": {"import_test_command": f"{basic_import}"},
            "[api]": {"import_test_command": f"{basic_import}"},
            "[complete-api]": {
                "import_test_command": f"{basic_import}; {s3_import}; {azure_blob_storage_import}; "
                f"{azure_key_vault_import}",
                "perform_vulnerability_check": True,
            },
            "[s3]": {"import_test_command": f"{basic_import}; {s3_import}"},
            "[azure-blob-storage]": {
                "import_test_command": f"{basic_import}; {azure_blob_storage_import}"
            },
            "[azure-key-vault]": {
                "import_test_command": f"{basic_import}; {azure_key_vault_import}"
            },
            "[alibaba-oss]": {"import_test_command": f"{basic_import}; {oss_import}"},
            "[google-cloud]": {
                "import_test_command": f"{basic_import}; {google_cloud_storage_import}; {google_cloud_bigquery_import}"
            },
            "[redis]": {"import_test_command": f"{basic_import}; {redis_import}"},
            "[kafka]": {"import_test_command": f"{basic_import}; {targets_import}"},
            "[complete]": {
                "import_test_command": f"{basic_import}; {s3_import}; {azure_blob_storage_import}; "
                + f"{azure_key_vault_import}; {google_cloud_storage_import};"
                + f" {redis_import}; {targets_import}; {oss_import}",
                "perform_vulnerability_check": True,
            },
            "[mlflow]": {"import_test_command": f"{basic_import}; {mlflow_import}"},
        }

    def run(self):
        self._logger.info("Running package tests in parallel")

        results = {}
        max_workers = min(len(self._extras_tests_data), os.cpu_count() or 4)
        self._logger.info(f"Using {max_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _test_extra_worker,
                    self._logger,
                    extra,
                    extra_tests_data,
                    self._package_installer,
                ): extra
                for extra, extra_tests_data in self._extras_tests_data.items()
            }

            for future in as_completed(futures):
                extra = futures[future]
                try:
                    result = future.result()
                    results[result["extra"]] = result["tests"]
                    self._logger.info(
                        f"Completed testing extra: {extra or 'base'}",
                        results=result["tests"],
                    )
                except Exception as exc:
                    self._logger.error(
                        f"Test for {extra} generated an exception", exc=exc
                    )
                    results[extra] = {
                        "import_test": {"passed": False},
                        "requirements_conflicts_test": {"passed": False},
                    }

        failed = False
        for extra, extra_tests_results in results.items():
            if not extra_tests_results.get("import_test", {}).get(
                "passed", False
            ) or not extra_tests_results.get("requirements_conflicts_test", {}).get(
                "passed", False
            ):
                failed = True
                break

        self._logger.info(
            "Finished running package tests", results=results, failed=failed
        )
        if failed:
            raise RuntimeError("Package tests failed")

    # Exposed funcaionlity for external test cases
    def test_requirements_vulnerabilities(self, extra):
        with Venv(self._logger, self._package_installer) as venv:
            _run_vulnerability_test(venv, extra, self._logger)


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
def run():
    logger = mlrun.utils.create_logger(
        level="debug",
        name="automation",
        formatter_kind=mlrun.utils.FormatterKinds.HUMAN_EXTENDED.name,
    )
    package_tester = PackageTester(logger)
    try:
        package_tester.run()
    except Exception as exc:
        logger.error("Failed running the package tester", exc=exc)
        raise


if __name__ == "__main__":
    main()
