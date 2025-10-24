#!/bin/bash
# Simple changelog updater for MLRun
# Usage: ./scripts/update-changelog.sh [--from TAG] [--version VERSION]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
FROM_TAG=""
VERSION=""
CHANGELOG_FILE="CHANGELOG.md"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_TAG="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --file)
            CHANGELOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--from TAG] [--version VERSION] [--file FILE]"
            echo ""
            echo "Options:"
            echo "  --from TAG       Start from this git tag (default: last tag)"
            echo "  --version VER    Set version for new entry (default: next patch)"
            echo "  --file FILE      Changelog file (default: CHANGELOG.md)"
            echo ""
            echo "Examples:"
            echo "  $0                          # Update with commits since last tag"
            echo "  $0 --from v1.2.0           # Update with commits since v1.2.0"  
            echo "  $0 --version 1.3.0         # Create entry for version 1.3.0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}📝 Updating Changelog${NC}"

# Get the range to check
if [[ -z "$FROM_TAG" ]]; then
    FROM_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [[ -z "$FROM_TAG" ]]; then
        echo -e "${YELLOW}No tags found, using all commits${NC}"
        COMMIT_RANGE="HEAD"
    else
        echo -e "${BLUE}Using commits since ${GREEN}$FROM_TAG${NC}"
        COMMIT_RANGE="$FROM_TAG..HEAD"
    fi
else
    echo -e "${BLUE}Using commits since ${GREEN}$FROM_TAG${NC}"  
    COMMIT_RANGE="$FROM_TAG..HEAD"
fi

# Get commits
commits=$(git log --oneline --no-merges $COMMIT_RANGE 2>/dev/null)

if [[ -z "$commits" ]]; then
    echo -e "${YELLOW}No new commits found${NC}"
    exit 0
fi

echo -e "${GREEN}Found $(echo "$commits" | wc -l) commits${NC}"

# Determine version
if [[ -z "$VERSION" ]]; then
    if [[ -n "$FROM_TAG" ]]; then
        # Extract version and increment patch
        current_version=${FROM_TAG#v}
        if command -v python3 >/dev/null 2>&1; then
            VERSION=$(python3 -c "
import re
v = '$current_version'
parts = v.split('.')
if len(parts) >= 3:
    parts[2] = str(int(parts[2]) + 1)
    print('.'.join(parts))
else:
    print(v + '.1')
")
        else
            # Fallback: just append .1 or increment last number
            if [[ $current_version =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
                major=${BASH_REMATCH[1]}
                minor=${BASH_REMATCH[2]}
                patch=${BASH_REMATCH[3]}
                VERSION="$major.$minor.$((patch + 1))"
            else
                VERSION="$current_version.1"
            fi
        fi
    else
        VERSION="0.1.0"
    fi
fi

echo -e "${BLUE}Creating entry for version ${GREEN}$VERSION${NC}"

# Categorize commits
added_items=""
changed_items=""
fixed_items=""
removed_items=""
other_items=""

while IFS= read -r commit; do
    # Extract commit message (remove hash)
    message=$(echo "$commit" | sed 's/^[a-f0-9]* //')
    
    # Categorize based on conventional commits or keywords
    case "$message" in
        feat:*|feat\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^feat[^:]*: *//')
            added_items="$added_items- $clean_msg"$'\n'
            ;;
        fix:*|fix\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^fix[^:]*: *//')
            fixed_items="$fixed_items- $clean_msg"$'\n'
            ;;
        remove:*|remove\(*\):*|rm:*|rm\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^r[me][^:]*: *//')
            removed_items="$removed_items- $clean_msg"$'\n'
            ;;
        docs:*|doc:*|docs\(*\):*|doc\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^docs\?[^:]*: *//')
            changed_items="$changed_items- $clean_msg"$'\n'
            ;;
        refactor:*|refactor\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^refactor[^:]*: *//')
            changed_items="$changed_items- $clean_msg"$'\n'
            ;;
        chore:*|chore\(*\):*)
            clean_msg=$(echo "$message" | sed 's/^chore[^:]*: *//')
            changed_items="$changed_items- $clean_msg"$'\n'
            ;;
        *fix*|*Fix*|*FIX*)
            fixed_items="$fixed_items- $message"$'\n'
            ;;
        *add*|*Add*|*ADD*|*new*|*New*)
            added_items="$added_items- $message"$'\n'
            ;;
        *remove*|*Remove*|*delete*|*Delete*)
            removed_items="$removed_items- $message"$'\n'
            ;;
        *)
            other_items="$other_items- $message"$'\n'
            ;;
    esac
done <<< "$commits"

# Build changelog entry
current_date=$(date +%Y-%m-%d)
new_entry="## [$VERSION] - $current_date"$'\n\n'

if [[ -n "$added_items" ]]; then
    new_entry="$new_entry### Added"$'\n'"$added_items"$'\n'
fi

if [[ -n "$changed_items" ]]; then
    new_entry="$new_entry### Changed"$'\n'"$changed_items"$'\n'
fi

if [[ -n "$fixed_items" ]]; then
    new_entry="$new_entry### Fixed"$'\n'"$fixed_items"$'\n'
fi

if [[ -n "$removed_items" ]]; then
    new_entry="$new_entry### Removed"$'\n'"$removed_items"$'\n'
fi

if [[ -n "$other_items" ]]; then
    new_entry="$new_entry### Other"$'\n'"$other_items"$'\n'
fi

# Add separator
new_entry="$new_entry---"$'\n\n'

# Update changelog file
if [[ ! -f "$CHANGELOG_FILE" ]]; then
    # Create new changelog
    cat > "$CHANGELOG_FILE" << EOF
# Changelog

All notable changes to this project will be documented in this file.

$new_entry
EOF
    echo -e "${GREEN}✓ Created new changelog file${NC}"
else
    # Insert after existing version (find first ## and insert before it)
    temp_file=$(mktemp)
    
    # Find the first version line and insert before it
    awk -v entry="$new_entry" '
    /^## \[[0-9]/ && !inserted {
        print entry
        inserted = 1
    }
    { print }
    END {
        if (!inserted) {
            print entry
        }
    }' "$CHANGELOG_FILE" > "$temp_file"
    
    mv "$temp_file" "$CHANGELOG_FILE"
    echo -e "${GREEN}✓ Updated changelog file${NC}"
fi

# Show what was added
echo -e "${BLUE}Added entry:${NC}"
echo "$new_entry" | head -20

# Show next steps
echo ""
echo -e "${GREEN}✓ Changelog updated successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review the changelog:"
echo "   cat $CHANGELOG_FILE"
echo "2. Edit manually if needed"
echo "3. Commit the changes:"
echo "   git add $CHANGELOG_FILE"
echo "   git commit -m 'docs: update changelog for v$VERSION'"