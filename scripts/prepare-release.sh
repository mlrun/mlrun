#!/bin/bash
# Release preparation script for MLRun
# Usage: ./scripts/prepare-release.sh --type [patch|minor|major] [--dry-run]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RELEASE_TYPE=""
DRY_RUN=false
CUSTOM_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            RELEASE_TYPE="$2"
            shift 2
            ;;
        --version)
            CUSTOM_VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --type [patch|minor|major|prerelease] [--version X.Y.Z] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --type TYPE       Release type: patch, minor, major, or prerelease"
            echo "  --version VERSION Custom version (overrides --type)"
            echo "  --dry-run         Show what would be done without making changes"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --type patch              # 1.2.3 → 1.2.4"
            echo "  $0 --type minor              # 1.2.3 → 1.3.0"
            echo "  $0 --type major              # 1.2.3 → 2.0.0"
            echo "  $0 --version 1.5.0           # Custom version"
            echo "  $0 --type minor --dry-run    # Preview changes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$CUSTOM_VERSION" && -z "$RELEASE_TYPE" ]]; then
    echo -e "${RED}Error: Must specify either --type or --version${NC}"
    exit 1
fi

if [[ -n "$RELEASE_TYPE" && ! "$RELEASE_TYPE" =~ ^(patch|minor|major|prerelease)$ ]]; then
    echo -e "${RED}Error: Invalid release type. Must be: patch, minor, major, or prerelease${NC}"
    exit 1
fi

# Check dependencies
command -v git >/dev/null 2>&1 || { echo -e "${RED}Error: git is required${NC}" >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo -e "${RED}Error: node is required for semver calculations${NC}" >&2; exit 1; }

# Install semver if not available
if ! command -v semver >/dev/null 2>&1; then
    echo -e "${YELLOW}Installing semver tool...${NC}"
    npm install -g semver
fi

echo -e "${BLUE}🚀 MLRun Release Preparation${NC}"
echo "================================"

# Get current version
echo -e "${BLUE}📋 Getting current version...${NC}"
current_version=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
current_version=${current_version#v}  # Remove 'v' prefix
echo -e "Current version: ${GREEN}$current_version${NC}"

# Calculate new version
if [[ -n "$CUSTOM_VERSION" ]]; then
    new_version="$CUSTOM_VERSION"
    echo -e "Using custom version: ${GREEN}$new_version${NC}"
else
    case "$RELEASE_TYPE" in
        patch)
            new_version=$(npx semver -i patch "$current_version")
            ;;
        minor)
            new_version=$(npx semver -i minor "$current_version")
            ;;
        major)
            new_version=$(npx semver -i major "$current_version")
            ;;
        prerelease)
            new_version=$(npx semver -i prerelease --preid=rc "$current_version")
            ;;
    esac
    echo -e "New ${RELEASE_TYPE} version: ${GREEN}$new_version${NC}"
fi

# Validate new version
if ! npx semver "$new_version" >/dev/null 2>&1; then
    echo -e "${RED}Error: Invalid version format: $new_version${NC}"
    exit 1
fi

if [[ "$new_version" == "$current_version" ]]; then
    echo -e "${RED}Error: New version same as current version${NC}"
    exit 1
fi

# Check if working directory is clean
echo -e "${BLUE}🔍 Checking working directory...${NC}"
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: Working directory not clean. Commit or stash changes first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Working directory is clean${NC}"

# Generate changelog
echo -e "${BLUE}📝 Generating changelog...${NC}"
if command -v conventional-changelog >/dev/null 2>&1; then
    # Use conventional-changelog if available
    changelog_content=$(conventional-changelog -p angular -s -r 2 | sed -n '/^## /,/^## /{/^## /!p;}' | head -n -1)
    if [[ -z "$changelog_content" ]]; then
        changelog_content="### Changed"$'\n'"- Various improvements and bug fixes"
    fi
else
    # Fallback: generate from git log
    echo -e "${YELLOW}conventional-changelog not found, generating basic changelog...${NC}"
    if [[ "$current_version" != "0.0.0" ]]; then
        commits=$(git log --oneline "v$current_version"..HEAD 2>/dev/null | head -20)
        if [[ -n "$commits" ]]; then
            changelog_content="### Changed"$'\n'"$(echo "$commits" | sed 's/^[a-f0-9]* /- /')"
        else
            changelog_content="### Changed"$'\n'"- Various improvements and bug fixes"
        fi
    else
        changelog_content="### Added"$'\n'"- Initial release"
    fi
fi

echo -e "${GREEN}✓ Generated changelog${NC}"
echo "Preview:"
echo "$changelog_content" | sed 's/^/  /'

# Update CHANGELOG.md
echo -e "${BLUE}📄 Updating CHANGELOG.md...${NC}"
current_date=$(date +%Y-%m-%d)
new_entry="## [$new_version] - $current_date"$'\n\n'"$changelog_content"$'\n'

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Would add to CHANGELOG.md:${NC}"
    echo "$new_entry" | sed 's/^/  /'
else
    if [[ -f "CHANGELOG.md" ]]; then
        # Insert after the [Unreleased] section
        temp_file=$(mktemp)
        awk -v entry="$new_entry" '/^## \[Unreleased\]/ {print; getline; print; print entry; next} 1' CHANGELOG.md > "$temp_file"
        mv "$temp_file" CHANGELOG.md
    else
        # Create new CHANGELOG.md
        cat > CHANGELOG.md << EOF
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

$new_entry
EOF
    fi
    echo -e "${GREEN}✓ Updated CHANGELOG.md${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}📋 Release Summary${NC}"
echo "=================="
echo -e "Version: ${GREEN}$current_version → $new_version${NC}"
echo -e "Type: ${GREEN}$RELEASE_TYPE${NC}"
echo -e "Date: ${GREEN}$current_date${NC}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}This was a dry run. No changes were made.${NC}"
    echo -e "${YELLOW}Run without --dry-run to apply changes.${NC}"
else
    echo ""
    echo -e "${GREEN}✓ Release prepared successfully!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Review the changes:"
    echo "   git diff"
    echo "2. Commit the changes:"
    echo "   git add CHANGELOG.md"
    echo "   git commit -m 'chore(release): prepare v$new_version'"
    echo "3. Create and push the tag:"
    echo "   git tag -a v$new_version -m 'Release v$new_version'"
    echo "   git push origin v$new_version"
    echo "4. Push the changes:"
    echo "   git push origin HEAD"
fi