import asyncio
import subprocess
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
    tags = loop.run_until_complete(_collect_image_tags(registry_url, images))
    for image, image_tags in tags.items():
        click.echo(f"Deleting {image} tags")
        for tag in image_tags:
            loop.run_until_complete(_delete_image_tag(registry_url, image, tag))

    click.echo("Removed all tags. Running garbage collection...")
    _run_registry_garbage_collection(registry_container_name)

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
