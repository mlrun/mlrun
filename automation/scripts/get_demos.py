# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import requests
from packaging.version import InvalidVersion, Version
from tqdm import tqdm

TIMEOUT = 30  # seconds
VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)(?:-rc\d+)?$")


def download_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise requests.RequestException
            sleep(2**attempt)  # Exponential backoff


def log(msg, repo):
    tqdm.write(f"[get_demos][{repo}] {msg}")


def download_release(repo, release_version):
    """Try to download the zip of a release tag."""
    url = f"https://github.com/{GITHUB_ORG}/{repo}/archive/refs/tags/{release_version}.tar.gz"
    log(f"Checking for release: {url}", repo)
    r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
    if r.status_code == 200:
        log(f"Found release {release_version}, downloading...", repo)
        archive_path = f"{repo}.tar.gz"
        try:
            resp = download_with_retry(url)
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            log(f"Extracting tar.gz to folder {DEST_DIR}", repo)

            with tarfile.open(archive_path, "r:gz") as tar:
                # Security: verify all members are within DEST_DIR
                for member in tar.getmembers():
                    member_path = os.path.abspath(os.path.join(DEST_DIR, member.name))
                    if not member_path.startswith(os.path.abspath(DEST_DIR)):
                        raise RuntimeError(
                            f"Attempted path traversal in tar: {member.name}"
                        )
                tar.extractall(DEST_DIR)

            return True

        except Exception as e:
            log(f"Download/extraction failed: {e}", repo)
            return False

        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)
    log(f"No release found for version ({release_version})", repo)
    return False


def remove_git_folder(repo):
    folder_path = DEST_DIR + "/" + repo + "/.github"
    log(f"Removing .github from folder {folder_path}", repo)
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        log("Warning: .github folder was not deleted", repo)


def rename_demo_folder(repo):
    # extracted_folders = [d for d in os.listdir(DEST_DIR) if d.startswith(f"{repo}-")]
    pattern = re.compile(rf"^{re.escape(repo)}-[\d\.]")
    extracted_folders = [d for d in os.listdir(DEST_DIR) if pattern.match(d)]
    if not extracted_folders:
        log(f"Warning: No extracted folder found matching pattern {repo}-*", repo)
        log("Warning: Demo folder name was not changed", repo)
        return
    if len(extracted_folders) > 1:
        log(f"Warning: Multiple folders found: {extracted_folders}, using first", repo)

    extracted_path = os.path.join(DEST_DIR, extracted_folders[0])
    final_path = os.path.join(DEST_DIR, repo)
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(extracted_path, final_path)


def get_all_releases(repo):
    url = f"https://api.github.com/repos/{GITHUB_ORG}/{repo}/releases"
    headers = {}
    # Support optional GitHub token
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(url, headers=headers, timeout=TIMEOUT)

    if response.status_code == 403:
        log("GitHub API rate limit exceeded", repo)
        raise RuntimeError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN environment variable."
        )

    if response.status_code == 200:
        releases = [r["tag_name"] for r in response.json()]
        return releases
    else:
        return []


def validate_versions(all_versions):
    valid_versions = []
    for v in all_versions:
        try:
            Version(v)
            valid_versions.append(v)
        except InvalidVersion:
            pass
    return valid_versions


def download_demo(demo_repo, mlrun_version):
    # Download exact given mlrun version
    log("Starting downloading process", demo_repo)
    all_releases = validate_versions(get_all_releases(demo_repo))
    if not all_releases:
        # No releases found in demo repository
        raise RuntimeError(
            f"Failed downloading {demo_repo}.\n"
            f"Repository {GITHUB_ORG}/{demo_repo} not exists or has no releases"
        )

    # Sorting versions
    sorted_versions = sorted(all_releases, key=Version, reverse=True)

    # Check if mlrun version is in the form of x.x.x or x.x.x-rcX
    match = VERSION_PATTERN.match(mlrun_version)
    if not match:
        log(
            f"Invalid mlrun version format: {mlrun_version}.\n"
            f"Check repository releases to find matching formats.\n"
            f"Using latest release {sorted_versions[0]}.",
            demo_repo,
        )

        return download_release(demo_repo, sorted_versions[0])

    # Removing -rcX if exists
    mlrun_version = match.group(1)

    # Finding all releases for demo that starts with the mlrun version
    matching_demo_releases = [r for r in sorted_versions if r.startswith(mlrun_version)]
    if matching_demo_releases:
        # Download the release or the latest rc for that mlrun version
        return download_release(demo_repo, mlrun_version) or download_release(
            demo_repo, max(matching_demo_releases)
        )

    log(
        f"Github repository has no matching release for mlrun version {mlrun_version}",
        demo_repo,
    )
    log("Using latest release", demo_repo)

    return download_release(demo_repo, sorted_versions[0])


def detect_demo_version(repo, mlrun_version):
    """Detect the downloaded version by querying GitHub releases API."""
    try:
        all_releases = validate_versions(get_all_releases(repo))
        if all_releases:
            sorted_versions = sorted(all_releases, key=Version, reverse=True)
            match = VERSION_PATTERN.match(mlrun_version)
            if match:
                base_version = match.group(1)
                matching_releases = [
                    r for r in sorted_versions if r.startswith(base_version)
                ]
                if matching_releases:
                    return matching_releases[0]
            return sorted_versions[0]
    except Exception:
        pass

    return "unknown"


def process_repo(repo, mlrun_version):
    """Process a demo repository and return its downloaded version."""
    try:
        if download_demo(repo, mlrun_version):
            rename_demo_folder(repo=repo)
            remove_git_folder(repo=repo)

            # Detect the downloaded version
            demo_version = detect_demo_version(repo, mlrun_version)

            log("Successfully processed", repo)
            return demo_version
        else:
            raise RuntimeError(
                f"Failed to download release from repository {GITHUB_ORG}/{repo}"
            )
    except Exception:
        # Cleanup on failure
        final_path = os.path.join(DEST_DIR, repo)
        if os.path.exists(final_path):
            shutil.rmtree(final_path, ignore_errors=True)
        raise


def create_manifest(mlrun_version, demo_versions):
    """Create a manifest file with version information for all downloaded demos."""
    manifest = {
        "mlrun_version": mlrun_version,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "github_org": GITHUB_ORG,
        "demos": demo_versions,
    }

    manifest_path = os.path.join(DEST_DIR, "demos_manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        tqdm.write(f"✅ Created manifest file: {manifest_path}")
    except Exception as e:
        tqdm.write(f"⚠️  Warning: Failed to create manifest file: {e}")


def get_demos(mlrun_version):
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"Configuration file not found: {CONFIG_PATH}")
    try:
        with Path(CONFIG_PATH).open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in configuration file {CONFIG_PATH}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file {CONFIG_PATH}: {e}")

    os.makedirs(DEST_DIR, exist_ok=True)
    repositories = config.get("demos")
    if not repositories:
        raise RuntimeError(f"No 'demos' key found in {CONFIG_PATH}")
    if not isinstance(repositories, list):
        raise RuntimeError(f"'demos' must be a list in {CONFIG_PATH}")
    if not all(isinstance(r, str) for r in repositories):
        raise RuntimeError(f"All demo entries must be strings in {CONFIG_PATH}")

    errors = []
    demo_versions = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_repo, repo, mlrun_version): repo
            for repo in repositories
        }
        with tqdm(
            total=len(futures),
            desc=f"Processing {len(futures)} repos",
            unit="repo",
            leave=True,  # Keep progress bar after completion
            position=0,  # Avoid multiple bars overlapping
        ) as pbar:
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    demo_version = future.result()
                    demo_versions[repo] = demo_version
                    pbar.set_postfix_str(f"✓ {repo}")
                except Exception as e:
                    errors.append((repo, e))
                    pbar.set_postfix_str(f"✗ {repo}")
                    tqdm.write(f"Error processing repo {repo}: {e}")
                pbar.update(1)

        if errors:
            error_details = "\n".join([f"  - {repo}: {str(e)}" for repo, e in errors])
            raise RuntimeError(
                f"Failed to process {len(errors)} out of {len(repositories)} repositories:\n{error_details}"
            )

        # Create manifest file
        create_manifest(mlrun_version, demo_versions)

        tqdm.write(
            f"\n✅ Successfully downloaded and processed all {len(repositories)} demos to '{DEST_DIR}/'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloading all demos using demos_config.json"
    )
    # Positional argument
    parser.add_argument(
        "mlrun_version",
        type=str,
        help="Demos version will be aligned with this version",
    )

    # Optional argument
    parser.add_argument("--org", default="mlrun", help="GitHub org")
    parser.add_argument(
        "--config_path", default="demos_config.json", help="Path to demos config file"
    )
    parser.add_argument(
        "--dest", default="demos", help="Folder name to extract demos to"
    )

    args = parser.parse_args()

    GITHUB_ORG = args.org
    CONFIG_PATH = args.config_path
    DEST_DIR = args.dest

    get_demos(args.mlrun_version)

    try:
        get_demos(args.mlrun_version)
        sys.exit(0)  # Explicit success
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)  # Explicit failure for CI/CD
