import subprocess
import sys

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
        google_cloud_storage_import = "import mlrun.datastore.google_cloud_storage"

        self._extras_tests_data = {
            "": {"import_test_command": f"{basic_import}"},
            "[api]": {
                "import_test_command": f"{basic_import}; {api_import}",
                "python_3.6_compatible": False,
            },
            "[complete-api]": {
                "import_test_command": f"{basic_import}; {api_import}; {s3_import}; "
                + f"{azure_blob_storage_import}; {azure_key_vault_import}",
                "python_3.6_compatible": False,
            },
            "[s3]": {"import_test_command": f"{basic_import}; {s3_import}"},
            "[azure-blob-storage]": {
                "import_test_command": f"{basic_import}; {azure_blob_storage_import}"
            },
            "[azure-key-vault]": {
                "import_test_command": f"{basic_import}; {azure_key_vault_import}"
            },
            "[google-cloud-storage]": {
                "import_test_command": f"{basic_import}; {google_cloud_storage_import}"
            },
            "[complete]": {
                "import_test_command": f"{basic_import}; {s3_import}; {azure_blob_storage_import}; "
                + f"{azure_key_vault_import}; {google_cloud_storage_import}",
            },
        }

    def run(self):
        self._logger.info("Running package tests",)

        results = {}
        for extra, extra_tests_data in self._extras_tests_data.items():
            if (
                sys.version_info[0] == 3
                and sys.version_info[1] == 6
                and not extra_tests_data.get("python_3.6_compatible", True)
            ):
                continue
            self._create_venv()
            self._install_extra(extra)
            results[extra] = {
                "import_test": {"passed": True},
                "requirements_conflicts_test": {"passed": True},
            }
            self._run_test(self._test_extra_imports, extra, results, "import_test")
            self._run_test(
                self._test_requirements_conflicts,
                extra,
                results,
                "requirements_conflicts_test",
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
            results[extra][test_key]["passed"] = False

    def _test_extra_imports(self, extra):
        self._logger.debug(
            "Testing extra imports", extra=extra,
        )
        test_command = (
            f"python -c '{self._extras_tests_data[extra]['import_test_command']}'"
        )
        self._run_command(
            test_command, run_in_venv=True,
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

    def _test_requirements_conflicts(self, extra):
        self._logger.debug(
            "Testing requirements conflicts", extra=extra,
        )
        self._run_command(
            "pip install pipdeptree", run_in_venv=True,
        )
        self._run_command(
            "pipdeptree --warn fail", run_in_venv=True,
        )

    def _create_venv(self):
        self._logger.debug("Creating venv",)
        self._run_command("python -m venv test-venv",)

    def _clean_venv(self):
        self._logger.debug("Cleaning venv",)
        self._run_command("rm -rf test-venv",)

    def _install_extra(self, extra):
        self._logger.debug(
            "Installing extra", extra=extra,
        )
        self._run_command(
            "python -m pip install --upgrade pip~=21.2.0", run_in_venv=True,
        )

        self._run_command(
            f"pip install '.{extra}'", run_in_venv=True,
        )

    def _run_command(self, command, run_in_venv=False, env=None):
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
def run():
    package_tester = PackageTester()
    try:
        package_tester.run()
    except Exception as exc:
        logger.error("Failed running the package tester", exc=exc)
        raise


if __name__ == "__main__":
    main()
