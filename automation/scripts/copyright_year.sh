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
#   ./copyright_year.sh fix               - fix copyright year in untracked files to the current year
#   ./copyright_year.sh check             - check copyright year in untracked files, exit 1 if wrong
#   ./copyright_year.sh check-ci <base>   - check copyright year in files added in a PR (CI use)

set -e

if [[ "$1" != "fix" && "$1" != "check" && "$1" != "check-ci" ]]; then
    echo "Usage: $0 {fix|check|check-ci <base-branch>}"
    exit 1
fi

current_year=$(date +%Y)
readonly COPYRIGHT_RE_OLD='Copyright 20[0-9][0-9] Iguazio'
readonly COPYRIGHT_LINE_CURRENT="Copyright $current_year Iguazio"

list_untracked_iguazio_copyright_paths() {
    local untracked
    untracked=$(git ls-files --others --exclude-standard)
    [ -z "$untracked" ] || echo "$untracked" | xargs grep -l "$COPYRIGHT_RE_OLD" 2>/dev/null || true
}

fail_if_bad_copyright() {
    # Usage: fail_if_bad_copyright <bad_files> <fix_hint>
    local bad_files="$1"
    local fix_hint="$2"
    if [ -n "$bad_files" ]; then
        echo "Wrong copyright year in new files (expected $current_year):"
        for f in $bad_files; do echo "  $f"; done
        echo "$fix_hint"
        exit 1
    fi
    echo "Copyright year check passed."
}

case "$1" in
    fix)
        copyright_files=$(list_untracked_iguazio_copyright_paths)
        if [ -n "$copyright_files" ]; then
            echo "$copyright_files" | xargs python -c \
                "import sys,re,fileinput; [print(re.sub('$COPYRIGHT_RE_OLD','$COPYRIGHT_LINE_CURRENT',line),end='') for line in fileinput.input(inplace=True)]"
        fi
        ;;
    check)
        copyright_files=$(list_untracked_iguazio_copyright_paths)
        bad_files=$([ -z "$copyright_files" ] || echo "$copyright_files" | xargs grep -L "$COPYRIGHT_LINE_CURRENT" 2>/dev/null || true)
        fail_if_bad_copyright "$bad_files" "Run 'make fmt' to fix automatically."
        ;;
    check-ci)
        base_branch="${2:?Usage: $0 check-ci <base-branch>}"
        bad_files=""
        for f in $(git diff --name-only --diff-filter=A "$base_branch"..HEAD); do
            if grep -q "$COPYRIGHT_RE_OLD" "$f" 2>/dev/null; then
                if ! grep -q "$COPYRIGHT_LINE_CURRENT" "$f" 2>/dev/null; then
                    bad_files="$bad_files $f"
                fi
            fi
        done
        fail_if_bad_copyright "$bad_files" "Update the copyright year to $current_year in the listed files, commit, and push. If the files are not yet committed, 'make fmt' fixes it automatically."
        ;;
esac
