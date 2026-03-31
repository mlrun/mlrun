#!/bin/bash
# Copyright 2026 Iguazio
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

# Manage copyright year in untracked (new) files.
# Usage:
#   ./copyright.sh fix               - fix copyright year in untracked files to the current year
#   ./copyright.sh check             - check copyright year in untracked files, exit 1 if wrong
#   ./copyright.sh check-ci <base>   - check copyright year in files added in a PR (CI use)

set -e

current_year=$(date +%Y)

case "$1" in
    fix)
        untracked=$(git ls-files --others --exclude-standard)
        copyright_files=$([ -z "$untracked" ] || echo "$untracked" | xargs grep -l "# Copyright 20[0-9][0-9] Iguazio" 2>/dev/null || true)
        if [ -n "$copyright_files" ]; then
            echo "$copyright_files" | xargs python -c \
                "import sys,re,fileinput; year=sys.argv.pop(1); [print(re.sub('# Copyright 20[0-9][0-9] Iguazio','# Copyright '+year+' Iguazio',line),end='') for line in fileinput.input(inplace=True)]" \
                "$current_year"
        fi
        ;;
    check)
        untracked=$(git ls-files --others --exclude-standard)
        copyright_files=$([ -z "$untracked" ] || echo "$untracked" | xargs grep -l "# Copyright 20[0-9][0-9] Iguazio" 2>/dev/null || true)
        bad_files=$([ -z "$copyright_files" ] || echo "$copyright_files" | xargs grep -L "# Copyright $current_year Iguazio" 2>/dev/null || true)
        if [ -n "$bad_files" ]; then
            echo "Wrong copyright year in new files (expected $current_year):"
            echo "$bad_files"
            echo "Run 'make fmt' to fix automatically."
            exit 1
        fi
        echo "Copyright year check passed."
        ;;
    check-ci)
        base_branch="${2:?Usage: $0 check-ci <base-branch>}"
        bad_files=""
        for f in $(git diff --name-only --diff-filter=A "$base_branch"..HEAD); do
            if grep -q "# Copyright 20[0-9][0-9] Iguazio" "$f" 2>/dev/null; then
                if ! grep -q "# Copyright $current_year Iguazio" "$f" 2>/dev/null; then
                    bad_files="$bad_files $f"
                fi
            fi
        done
        if [ -n "$bad_files" ]; then
            echo "Wrong copyright year in new files (expected $current_year):"
            for f in $bad_files; do echo "  $f"; done
            echo "Update the copyright year to $current_year in the listed files, commit, and push. If the files are not yet committed, 'make fmt' fixes it automatically."
            exit 1
        fi
        echo "Copyright year check passed."
        ;;
    *)
        echo "Usage: $0 {fix|check|check-ci <base-branch>}"
        exit 1
        ;;
esac
