import re
import sys

TARGET_MODULES = {"mlrun", "mlrun_pipelines"}

DISALLOWED_PATTERN = re.compile(r"^\s*from\s+(mlrun|mlrun_pipelines)\s+import")


def check_disallowed_imports(file_path: str):
    """
    Check for disallowed imports in a file and return a list of violations.
    """
    violations = []
    with open(file_path) as file:
        for line_num, line in enumerate(file, start=1):
            if DISALLOWED_PATTERN.match(line.strip()):
                violations.append(f"{file_path}:{line_num}: Disallowed import '{line.strip()}'."
                                  f" Use 'import {line.split()[1]}' instead.")
    return violations


def main(files: list[str]):
    """
    Scan all provided files and report disallowed imports.
    """
    all_violations = []
    for file_path in files:
        if file_path.endswith(".py"):
            all_violations.extend(check_disallowed_imports(file_path))

    if all_violations:
        print("The following disallowed imports were found:")
        for violation in all_violations:
            print(f"❌ {violation}")
        sys.exit(1)
    else:
        print("✅ All imports are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
