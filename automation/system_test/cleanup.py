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

import asyncio
import subprocess
import time
import typing

import aiohttp
import click


@click.group()
def main():
    pass


@main.command()
@click.argument("registry-url", type=str, required=True)
@click.argument("registry-container-name", type=str, required=True)
@click.argument(
    "images",
    type=str,
    required=True,
)
def docker_images(registry_url: str, registry_container_name: str, images: str):
    images = images.split(",")
    loop = asyncio.get_event_loop()
    try:
        click.echo("Removing images from datanode docker")
        _remove_image_from_datanode_docker()
    except Exception as exc:
        click.echo(
            f"Unable to remove images from datanode docker: {exc}, continuing anyway"
        )

    try:
        click.echo("Removing dangling images from datanode docker")
        _remove_dangling_images_from_datanode_docker()
    except Exception as exc:
        click.echo(
            f"Unable to remove dangling images from datanode docker: {exc}, continuing anyway"
        )

    try:
        _run_registry_garbage_collection(registry_container_name)
    except Exception as exc:
        click.echo(f"Unable to run garbage collection: {exc}, continuing anyway")

    # to make sure everything is up-to-date in the registry cache
    click.echo("Restarting datanode docker registry and sleeping for 30 seconds")
    _restart_docker_registry(registry_container_name)

    # give the registry some time to start
    time.sleep(30)
    tags = loop.run_until_complete(_collect_image_tags(registry_url, images))
    loop.run_until_complete(_delete_image_tags(registry_url, tags))

    click.echo("Removed all tags. Running garbage collection...")
    _run_registry_garbage_collection(registry_container_name)

    click.echo("Restarting datanode docker registry...")
    _restart_docker_registry(registry_container_name)

    click.echo("Cleaning images from local Docker cache...")
    _clean_images_from_local_docker_cache(tags)


async def _collect_image_tags(
    registry: str, images: typing.List[str]
) -> typing.Dict[str, typing.List[str]]:
    """Collect all image tags from Docker Hub."""
    tags = {}
    async with aiohttp.ClientSession() as session:
        for image in images:
            async with session.get(f"{registry}/v2/{image}/tags/list") as response:
                if response.status != 200:
                    click.echo(
                        f"Unable to fetch tags for {image}: {response.status}, skipping"
                    )
                    continue
                data = await response.json()
                if data.get("tags"):
                    tags[image] = data["tags"]
    return tags


def _remove_image_from_datanode_docker():
    """Remove image from datanode docker"""
    formatted_docker_images = subprocess.Popen(
        ["docker", "images", "--format", "'{{.Repository }}:{{.Tag}}'"],
        stdout=subprocess.PIPE,
    )
    grep = subprocess.Popen(
        ["grep", "mlrun"],
        stdin=formatted_docker_images.stdout,
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["xargs", "--no-run-if-empty", "docker", "rmi", "-f"],
        stdin=grep.stdout,
    )
    formatted_docker_images.stdout.close()
    grep.stdout.close()


def _remove_dangling_images_from_datanode_docker():
    """Remove dangling images from datanode docker"""

    dangling_docker_images = subprocess.Popen(
        ["docker", "images", "--quiet", "--filter", "dangling=true"],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["xargs", "--no-run-if-empty", "docker", "rmi", "-f"],
        stdin=dangling_docker_images.stdout,
    )
    dangling_docker_images.stdout.close()


async def _delete_image_tags(
    registry: str, tags: typing.Dict[str, typing.List[str]]
) -> None:
    for image, image_tags in tags.items():
        click.echo(f"Deleting {image} tags")
        for tag in image_tags:
            try:
                await _delete_image_tag(registry, image, tag)
            except Exception as exc:
                click.echo(f"Unable to delete {image}:{tag}: {exc}")


async def _delete_image_tag(registry: str, image: str, tag: str) -> None:
    """Delete a single image tag."""
    digest = await _get_tag_digest(registry, image, tag)
    async with aiohttp.ClientSession() as session:
        click.echo(f"\tDeleting {image}:{tag} ({digest})")
        async with session.delete(
            f"{registry}/v2/{image}/manifests/{digest}"
        ) as response:
            if response.status != 202:
                raise RuntimeError(f"Unable to delete {image}:{tag}: {response.status}")


async def _get_tag_digest(registry: str, image: str, tag: str) -> str:
    """Get the digest for a single image tag."""
    async with aiohttp.ClientSession() as session:
        async with session.head(
            f"{registry}/v2/{image}/manifests/{tag}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Unable to fetch digest for {image}:{tag}: {response.status}"
                )
            return response.headers["Docker-Content-Digest"]


def _run_registry_garbage_collection(registry_container_name: str) -> None:
    """Run Docker registry garbage collection."""
    subprocess.run(
        [
            "docker",
            "exec",
            registry_container_name,
            "registry",
            "garbage-collect",
            "--delete-untagged=true",
            "/etc/docker/registry/config.yml",
        ]
    )


def _restart_docker_registry(registry_container_name: str) -> None:
    """Restart Docker registry."""
    subprocess.run(["docker", "restart", registry_container_name])


def _clean_images_from_local_docker_cache(
    tags: typing.Dict[str, typing.List[str]]
) -> None:
    """Clean images from local Docker cache."""
    command = ["docker", "rmi", "-f"]
    command.extend(
        [f"{image}:{tag}" for image, image_tags in tags.items() for tag in image_tags]
    )
    subprocess.run(command)


if __name__ == "__main__":
    main()
