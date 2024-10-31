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

import multiprocessing
import typing

from mlrun.utils import logger


def evaluate_condition_in_separate_process(
    condition: str, context: dict[str, typing.Any], timeout: int = 5
):
    if not condition:
        return True

    receiver, sender = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=_evaluate_condition_wrapper,
        args=(sender, condition, context),
    )
    p.start()
    if receiver.poll(timeout):
        result = receiver.recv()
        p.join()
        return result
    else:
        p.kill()
        logger.warning(
            f"Condition evaluation timed out after {timeout} seconds. Ignoring condition",
            condition=condition,
        )
        return True


def _evaluate_condition_wrapper(
    connection, condition: str, context: dict[str, typing.Any]
):
    connection.send(_evaluate_condition(condition, context))
    return connection.close()


def _evaluate_condition(condition: str, context: dict[str, typing.Any]):
    import jinja2.sandbox

    jinja_env = jinja2.sandbox.SandboxedEnvironment()
    template = jinja_env.from_string(condition)
    result = template.render(**context)
    if result.lower() in ["0", "no", "n", "f", "false", "off"]:
        return False

    # if the condition is not a boolean, we ignore the condition
    return True
