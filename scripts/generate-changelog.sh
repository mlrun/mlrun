#!/bin/bash
# Changelog generation helper for MLRun
# Usage: ./scripts/generate-changelog.sh [--from TAG] [--format FORMAT]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
FROM_TAG=""
FORMAT="conventional"
OUTPUT_FILE=""
INCLUDE_UNRELEASED=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_TAG="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --no-unreleased)
            INCLUDE_UNRELEASED=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--from TAG] [--format FORMAT] [--output FILE] [--no-unreleased]"
            echo ""
            echo "Options:"
            echo "  --from TAG        Generate changelog from this tag to HEAD"
            echo "  --format FORMAT   Output format: conventional (default), simple, or full"
            echo "  --output FILE     Write to file instead of stdout"
            echo "  --no-unreleased  Don't include unreleased changes"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # All unreleased changes"
            echo "  $0 --from v1.2.0               # Changes since v1.2.0"
            echo "  $0 --format simple             # Simple format"
            echo "  $0 --output CHANGELOG.tmp       # Write to file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}📝 MLRun Changelog Generator${NC}" >&2
echo "==============================" >&2

# Determine range
if [[ -z "$FROM_TAG" ]]; then
    # Get latest tag
    FROM_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [[ -z "$FROM_TAG" ]]; then
        echo -e "${YELLOW}No tags found, showing all commits${NC}" >&2
        COMMIT_RANGE="HEAD"
    else
        echo -e "${BLUE}Generating changelog from ${GREEN}$FROM_TAG${BLUE} to ${GREEN}HEAD${NC}" >&2
        COMMIT_RANGE="$FROM_TAG..HEAD"
    fi
else
    echo -e "${BLUE}Generating changelog from ${GREEN}$FROM_TAG${BLUE} to ${GREEN}HEAD${NC}" >&2
    COMMIT_RANGE="$FROM_TAG..HEAD"
fi

# Get commits
commits=$(git log --oneline --no-merges $COMMIT_RANGE 2>/dev/null)

if [[ -z "$commits" ]]; then
    echo -e "${YELLOW}No commits found in range${NC}" >&2
    exit 0
fi

# Count commits by type
feat_count=$(echo "$commits" | grep -c "^[a-f0-9]* feat" || echo "0")
fix_count=$(echo "$commits" | grep -c "^[a-f0-9]* fix" || echo "0")
chore_count=$(echo "$commits" | grep -c "^[a-f0-9]* chore" || echo "0")
docs_count=$(echo "$commits" | grep -c "^[a-f0-9]* docs" || echo "0")
refactor_count=$(echo "$commits" | grep -c "^[a-f0-9]* refactor" || echo "0")
test_count=$(echo "$commits" | grep -c "^[a-f0-9]* test" || echo "0")
other_count=$(echo "$commits" | grep -cv "^[a-f0-9]* \(feat\|fix\|chore\|docs\|refactor\|test\)" || echo "0")

echo -e "${BLUE}Found commits: ${GREEN}feat($feat_count) fix($fix_count) chore($chore_count) docs($docs_count) refactor($refactor_count) test($test_count) other($other_count)${NC}" >&2

# Generate changelog based on format
generate_changelog() {
    case "$FORMAT" in
        conventional)
            # Group by conventional commit types
            if [[ $feat_count -gt 0 ]]; then
                echo "### ✨ Features"
                echo "$commits" | grep "^[a-f0-9]* feat" | sed 's/^[a-f0-9]* feat\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* feat: /- /'
                echo ""
            fi
            
            if [[ $fix_count -gt 0 ]]; then
                echo "### 🐛 Bug Fixes"
                echo "$commits" | grep "^[a-f0-9]* fix" | sed 's/^[a-f0-9]* fix\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* fix: /- /'
                echo ""
            fi
            
            if [[ $refactor_count -gt 0 ]]; then
                echo "### ♻️ Refactoring"
                echo "$commits" | grep "^[a-f0-9]* refactor" | sed 's/^[a-f0-9]* refactor\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* refactor: /- /'
                echo ""
            fi
            
            if [[ $docs_count -gt 0 ]]; then
                echo "### 📚 Documentation"
                echo "$commits" | grep "^[a-f0-9]* docs" | sed 's/^[a-f0-9]* docs\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* docs: /- /'
                echo ""
            fi
            
            if [[ $test_count -gt 0 ]]; then
                echo "### 🧪 Tests"
                echo "$commits" | grep "^[a-f0-9]* test" | sed 's/^[a-f0-9]* test\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* test: /- /'
                echo ""
            fi
            
            if [[ $chore_count -gt 0 ]]; then
                echo "### 🔧 Chores"
                echo "$commits" | grep "^[a-f0-9]* chore" | sed 's/^[a-f0-9]* chore\(([^)]*)\)\?: /- /' | sed 's/^[a-f0-9]* chore: /- /'
                echo ""
            fi
            
            if [[ $other_count -gt 0 ]]; then
                echo "### 📦 Other Changes"
                echo "$commits" | grep -v "^[a-f0-9]* \(feat\|fix\|chore\|docs\|refactor\|test\)" | sed 's/^[a-f0-9]* /- /'
                echo ""
            fi
            ;;
            
        simple)
            echo "### Changes"
            echo "$commits" | sed 's/^[a-f0-9]* /- /'
            echo ""
            ;;
            
        full)
            echo "### Detailed Changes"
            echo ""
            while IFS= read -r commit; do
                hash=$(echo "$commit" | cut -d' ' -f1)
                message=$(echo "$commit" | cut -d' ' -f2-)
                author=$(git show --no-patch --format='%an' "$hash")
                date=$(git show --no-patch --format='%ad' --date=short "$hash")
                echo "- **$message** ($date) - $author"
            done <<< "$commits"
            echo ""
            ;;
    esac
}

# Output
changelog_content=$(generate_changelog)

if [[ -n "$OUTPUT_FILE" ]]; then
    echo "$changelog_content" > "$OUTPUT_FILE"
    echo -e "${GREEN}✓ Changelog written to $OUTPUT_FILE${NC}" >&2
else
    echo "$changelog_content"
fi

echo -e "${GREEN}✓ Changelog generated successfully${NC}" >&2