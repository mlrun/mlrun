#!/usr/bin/env python3
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
#

import argparse
import os
import subprocess
from typing import List, Union


class MlRunReplacer(object):
    def __init__(self, args):
        self._verbose = args.verbose

    def _exec_cmd(self, cmd: List[str]) -> str:
        if self._verbose:
            print(" ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def _exec_silent_cmd(self, cmd: List[str]):
        if self._verbose:
            print(" ".join(cmd))
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)

    def _get_mlrun_api_pod(self) -> Union[str, None]:
        print("Get mlrun-api pod name")
        cmd = ["kubectl", "-n", "default-tenant", "get", "pod"]

        for line in self._exec_cmd(cmd).splitlines()[1:]:
            if "mlrun-api-chief" in line:
                return line.split()[0]

    def _get_mlrun_api_image(self, pod_name: str) -> Union[str, None]:
        print("Get mlrun-api image name")
        cmd = ["kubectl", "-n", "default-tenant", "get", "pods", pod_name, "-o", "yaml"]

        for line in self._exec_cmd(cmd).splitlines():
            if "image" in line and "mlrun-api" in line:
                return line.split("image:")[1].strip()

    def _build_api_docker(self, version: str) -> Union[str, None]:
        os.environ["MLRUN_VERSION"] = version
        image_tag_str = "Successfully tagged "

        print("Build mlrun-api image")

        cmd = ["make", "api"]

        for line in self._exec_cmd(cmd).splitlines():
            if image_tag_str in line and "mlrun-api" in line:
                return line.split(image_tag_str)[-1].strip()

    def _replace_docker_tag(self, new_image: str, orig_image: str):
        print("Replace docker tag")
        cmd = ["docker", "tag", new_image, orig_image]
        self._exec_silent_cmd(cmd)

    def _delete_orig_pod(self):
        print("Delete dockers, so that new images will replace them")
        cmd = [
            "kubectl",
            "-n",
            "default-tenant",
            "delete",
            "pod",
            "-l",
            "app.kubernetes.io/instance=mlrun",
            "-l",
            "app.kubernetes.io/component=api",
        ]
        self._exec_silent_cmd(cmd)

    def do_replacing(self):
        api_pod = self._get_mlrun_api_pod()
        if not api_pod:
            raise RuntimeError(
                "mlrun-api pod name not found. perhaps you are running on old version without chief pod"
            )

        api_image = self._get_mlrun_api_image(api_pod)
        if not api_image:
            raise RuntimeError("mlrun-api image name not found")

        api_image_version = api_image.split("mlrun-api:")[-1].strip()

        if self._verbose:
            print(f"Version: {api_image_version} from {api_image}")

        new_image_name = self._build_api_docker(api_image_version)
        if not new_image_name:
            raise RuntimeError("unable to build api docker")

        self._replace_docker_tag(new_image_name, api_image)
        self._delete_orig_pod()
        print("Replaced mlrun-api image sucessfully! Yay!")


def main():
    parser = argparse.ArgumentParser(description="Replace mlrun")

    parser.add_argument(
        "--verbose", help="Print what we are doing", action="store_true"
    )

    args = parser.parse_args()

    MlRunReplacer(args).do_replacing()


if __name__ == "__main__":
    main()
