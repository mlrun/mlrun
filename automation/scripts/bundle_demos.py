# Copyright 2019 Iguazio
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
#
import os
import shutil
import tarfile
import traceback

import requests
from git import Repo

# List of repositories to archive
repos = [
    {"url": "https://github.com/mlrun/demo-azure-ML.git", "name": "demo-azure-ML"},
    {"url": "https://github.com/mlrun/demo-fraud.git", "name": "demo-fraud"},
    {
        "url": "https://github.com/mlrun/demo-mask-detection.git",
        "name": "demo-mask-detection",
    },
    # Add more repositories as needed
]
demos_files = {
    "heart": [  # azure-ml
        "https://s3.us-east-1.wasabisys.com/iguazio/data/heart/heart.csv"
    ],
    "fraud-demo-mlrun-fs-docs": [  # fraud
        "https://s3.us-east-1.wasabisys.com/iguazio/data/fraud-demo-mlrun-fs-docs/user_events.csv",
        "https://s3.us-east-1.wasabisys.com/iguazio/data/fraud-demo-mlrun-fs-docs/events.csv",
        "https://s3.us-east-1.wasabisys.com/iguazio/data/fraud-demo-mlrun-fs-docs/data.csv",
    ],
    "prajnasb-generated-mask-detection": [  # mask detection
        "https://s3.us-east-1.wasabisys.com/iguazio/data/prajnasb-generated-mask-detection/prajnasb_generated_mask_detection.zip"
    ],
}

# Directory where repositories will be cloned and extracted
temp_dir = "demos"

# Create the temporary directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Clone each repository and extract its contents to the temporary directory
for repo_info in repos:
    print(f"cloning {repo_info['url']} with name {repo_info['name']} to {temp_dir}")
    repo_url = repo_info["url"]
    repo_name = repo_info["name"]

    # Clone the repository
    try:
        repo = Repo.clone_from(repo_url, os.path.join(temp_dir, repo_name))
    except Exception as e:
        print(f"could not clone repo {repo_url}")
        print(e)

# Copying update_demos.sh from mlrun
shutil.copy2("automation/scripts/update_demos.sh", "demos/update_demos.sh")

for demo, files in demos_files.items():
    print(f"Downloading {len(files)} files for demo {demo}...")
    for url in files:
        file_name = os.path.join(temp_dir, "data", demo, url.split("/")[-1])
        os.makedirs(os.path.join(temp_dir, "data", demo), exist_ok=True)
        print(f"Downloading {url}...")
        # Download the file and save it
        try:
            response = requests.get(url)
            with open(file_name, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"could not download {url}")
            print(e)
            traceback.print_exc()

# Create a tar archive of the temporary directory
with tarfile.open("mlrun-demos.tar", "w") as tar:
    tar.add(temp_dir, arcname=os.path.basename(temp_dir))

print("Archive created successfully!")

# Cleanup: Delete the temporary directory
shutil.rmtree(temp_dir)
