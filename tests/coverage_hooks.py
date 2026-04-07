# # Copyright 2026 Iguazio
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# # ---------------------------------------------------------------------------
# # Forked-process coverage support
# # Pytest-forked isolates test execution in new processes, which can prevent
# # coverage from automatically attaching to the child. We use the
# # COVERAGE_PROCESS_START environment variable to force the tracer to
# # initialize in the forked process, ensuring that coverage data is captured
# # correctly across process boundaries.
# # ---------------------------------------------------------------------------
#
# import os
# import sys
#
#
# def is_coverage_active():
#     """Return True if coverage is already tracing."""
#     import coverage
#
#     return isinstance(
#         getattr(sys, "gettrace", lambda: None)(), coverage.Coverage.__class__
#     )
#
#
# def pytest_runtest_call(item):
#     """If running inside a forked child process, ensure coverage is active."""
#     if os.environ.get("COVERAGE_PROCESS_START") and not is_coverage_active():
#         import coverage
#
#         cov = coverage.Coverage(
#             config_file=os.environ["COVERAGE_PROCESS_START"],
#             data_file=os.environ.get("COVERAGE_FILE"),
#             data_suffix=True,
#         )
#         cov.start()
#         item._coverage_forked = cov
#
#
# def pytest_runtest_teardown(item, nextitem):
#     """Stop and save coverage that was started in forked child."""
#     cov = getattr(item, "_coverage_forked", None)
#     if cov is not None:
#         cov.stop()
#         # Save the data only if there is some; otherwise we might get a "No data was collected" error
#         if cov._collector and cov._collector.data:
#             cov.save()
