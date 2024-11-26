import re
import sys

TARGET_MODULES = {"mlrun", "mlrun_pipelines"}


def check_disallowed_imports(file_path):
    violations = []
    with open(file_path) as file:
        for line_num, line in enumerate(file, start=1):
            stripped_line = line.strip()

            # Match disallowed imports: 'from mlrun import x' or 'from mlrun_pipelines import x'
            disallowed_pattern = r"^\s*from\s+(mlrun|mlrun_pipelines)\s+import"

            if re.match(disallowed_pattern, stripped_line):
                violations.append(
                    f"{file_path}:{line_num}: Disallowed import '{stripped_line}'."
                    f"Use 'import mlrun' or 'import mlrun_pipelines' instead."
                )
    return violations


if __name__ == "__main__":
    files = sys.argv[1:]
    all_violations = []
    for file_path in files:
        if file_path.endswith(".py"):
            violations = check_disallowed_imports(file_path)
            all_violations.extend(violations)

    if all_violations:
        print("The following disallowed imports were found:")
        for violation in all_violations:
            print(f"❌ {violation}")
        sys.exit(1)
    else:
        print("✅ All imports are valid.")
