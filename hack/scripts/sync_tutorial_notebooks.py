#!/usr/bin/env python

import argparse
import os
import os.path
import subprocess


def main(_args):
    # this is intended to be run from the repo root
    repo_root = os.getcwd()

    mlrun_demos_ref = args.mlrun_demos_ref
    tutorials_dir = os.path.join(repo_root, "docs", "getting-started-tutorial")
    tutorials_images_dir = os.path.join(
        repo_root, "docs", "_static", "getting-started-tutorial"
    )

    # tuples (source path, dest dir)
    # the files will move to dest dir but remain with the same name
    files_to_transplant = [
        ("01-mlrun-basics.ipynb", tutorials_dir),
        ("02-model-training.ipynb", tutorials_dir),
        ("03-model-serving.ipynb", tutorials_dir),
        ("04-pipeline.ipynb", tutorials_dir),
        ("images/func-schedule.jpg", tutorials_images_dir),
        ("images/jobs.png", tutorials_images_dir),
        ("images/kubeflow-pipeline.png", tutorials_images_dir),
        ("images/nuclio-deploy.png", tutorials_images_dir),
    ]
    print(
        f"Fetching tutorial notebooks and assets from mlrun/demos ref={mlrun_demos_ref} ..."
    )
    os.makedirs(os.path.join(tutorials_images_dir, "images"), exist_ok=True)
    for source_file, dest_path in files_to_transplant:
        cmd = (
            f"curl -sSL "
            f"https://raw.githubusercontent.com/mlrun/demos/{mlrun_demos_ref}/getting-started-tutorial/{source_file} "
            f"--output {dest_path}/{source_file}"
        )

        subprocess.check_call(cmd, shell=True)

    # escape the url (similar to MLRUN_DEMOS_REF but with a slight change "release/" -> "release-"
    print(f"Fixing files in {tutorials_dir} ...")
    mlrun_demos_ref_changes = mlrun_demos_ref.replace(r"release/", "release-")

    base_docs_url = f"https://mlrun.readthedocs.io/en/{mlrun_demos_ref_changes}"
    for (file_candidate, file_dir) in files_to_transplant:
        if not file_candidate.endswith(".ipynb"):
            continue

        file_path = os.path.join(tutorials_dir, file_candidate)
        with open(file_path, "r") as file:
            file_contents = file.read()

        # replace the docs www url to relative local urls
        file_contents = file_contents.replace(base_docs_url, "..")

        # replace images files dir with new dest
        file_contents = file_contents.replace(
            "./images", f"{tutorials_images_dir}/images"
        )

        # replace .html extension with .md
        file_contents = file_contents.replace(".html", ".md")

        # write the file out again
        with open(file_path, "w") as file:
            file.write(file_contents)
        print(f"Edited {file_path}")

    print("Done")


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlrun-demos-ref", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    main(args)
