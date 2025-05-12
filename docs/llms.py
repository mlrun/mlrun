import json
import os
import re

EXCLUDE_DIRS = {"_build", ".git", "venv"}
BASE_DOCS_URL = "https://docs.mlrun.org/en/stable/"


def generate_llm_txt(root_dir, prefix="", output_path=None, exclude_dirs=None):
    """
    Generates llms.txt by categorizing .md and .ipynb files with extracted titles and descriptions.
    """
    exclude_dirs = exclude_dirs or set()
    markdown_files = find_files_by_extension(
        root_dir, [".md"], exclude_dirs=exclude_dirs
    )
    notebook_files = find_files_by_extension(
        root_dir, [".ipynb"], exclude_dirs=exclude_dirs
    )

    parsed_output = []

    # Add Markdown files
    add_files_to_output(
        markdown_files,
        root_dir,
        parsed_output,
        "# Documentation",
        extract_md_first_title,
        extract_md_first_sentence,
    )

    # Add Notebook files
    add_files_to_output(
        notebook_files,
        root_dir,
        parsed_output,
        "\n## Examples",
        extract_ipynb_first_title,
        extract_ipynb_first_sentence,
    )

    # Determine output path
    output_path = output_path or os.path.join(root_dir, "llms.txt")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as file:
        formatted_output = "\n".join(parsed_output)
        file.write(f"{prefix}{formatted_output}")

    print(f"✅ Generated llms.txt in {output_path}")


def find_files_by_extension(root_dir, extensions, exclude_dirs=None):
    """
    Recursively find all files with given extensions, excluding specified directories.
    """
    exclude_dirs = exclude_dirs or set()
    found_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(excluded in dirpath for excluded in exclude_dirs):
            continue  # Skip excluded directories

        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                found_files.append(os.path.join(dirpath, filename))

    return found_files


def add_files_to_output(
    files, root_dir, parsed_output, section, title_extractor, sentence_extractor
):
    """
    Helper function to add files to parsed_output with title and description.
    """
    parsed_output.append(section)
    for file in files:
        title = title_extractor(file)
        description = sentence_extractor(file)
        if title:
            relative_path = os.path.relpath(file, root_dir)
            html_path = os.path.splitext(relative_path)[0] + ".html"
            full_url = f"{BASE_DOCS_URL}{html_path.replace(os.sep, '/')}"
            print(full_url)
            parsed_output.append(f"- [{title}]({full_url}): {description}")


def extract_md_first_title(md_path):
    """
    Extract the first level-1 title (# Title) from a Markdown file.
    """
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^#\s+(.+)", line.strip())
            if match:
                # Return the title without any extra formatting or HTML comments
                return re.sub(
                    r"\s*<!--.*?-->", "", match.group(1), flags=re.DOTALL
                ).strip()
    raise Exception(f"No title found in file {md_path}")


def extract_md_first_sentence(md_path):
    """
    Extract the first meaningful sentence or paragraph after the title from a Markdown file.
    """
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if result := valid_first_sentence(line):
            return result
    raise Exception(f"No description found in file {md_path}")


def extract_ipynb_first_title(nb_path):
    """
    Extract the first Markdown cell's first line as the title from a Jupyter Notebook.
    """
    with open(nb_path, encoding="utf-8") as file:
        notebook = json.load(file)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            lines = cell.get("source", [])
            if isinstance(lines, str):
                # if the source is a single string, convert it into a list of strings
                lines = [lines]
            for line in lines:
                line = line.strip()
                if line.startswith("# "):
                    return re.sub(
                        r"\s*<!--.*?-->", "", line[2:], flags=re.DOTALL
                    ).strip()
    raise Exception(f"No title found in file {nb_path}")


def extract_ipynb_first_sentence(nb_path):
    """
    Extracts the first meaningful sentence or paragraph after the title
    from the first Markdown cell in a Jupyter Notebook.
    """
    with open(nb_path, encoding="utf-8") as file:
        notebook = json.load(file)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            for line in cell.get("source", []):
                if result := valid_first_sentence(line):
                    return result
    raise Exception(f"No description found in file {nb_path}")


def valid_first_sentence(line):
    line = line.strip()
    # Skip title lines and empty lines
    if line.startswith("#") or line.startswith("<"):
        return None
    if re.match(r"^\(.*\)=", line):
        # Skip anchor-like metadata lines
        return None
    if line:
        # After skipping header lines
        # Extract first sentence after header
        return line.split(".")[0] + "."
