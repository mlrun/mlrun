#!/bin/bash
# Get previous version using git history
# Usage: ./scripts/get-previous-version.sh [method]

METHOD=${1:-"git-tags"}

case "$METHOD" in
    "git-tags")
        echo "🏷️ Finding previous version from git tags..."
        # Get the second-to-last version tag
        PREVIOUS_VERSION=$(git tag -l 'v*.*.*' --sort=-version:refname | head -n2 | tail -n1)
        if [[ -z "$PREVIOUS_VERSION" ]]; then
            PREVIOUS_VERSION=$(git tag --sort=-version:refname | head -n2 | tail -n1)
        fi
        ;;
        
    "bumpversion-commits")
        echo "📝 Finding previous version from bumpversion commits..."
        # Look for the last bumpversion commit
        PREVIOUS_COMMIT=$(git log --oneline --grep="Bump version" | head -n2 | tail -n1 | cut -d' ' -f1)
        if [[ -n "$PREVIOUS_COMMIT" ]]; then
            PREVIOUS_VERSION=$(git show "$PREVIOUS_COMMIT:.bumpversion.cfg" | grep "current_version" | cut -d'=' -f2 | xargs)
        fi
        ;;
        
    "config-file")
        echo "📄 Reading from version info file..."
        if [[ -f ".previous_version" ]]; then
            PREVIOUS_VERSION=$(cat .previous_version)
        elif [[ -f "VERSION_INFO.txt" ]]; then
            PREVIOUS_VERSION=$(grep "previous_version" VERSION_INFO.txt | cut -d'=' -f2)
        fi
        ;;
        
    "changelog")
        echo "📰 Finding previous version from CHANGELOG..."
        if [[ -f "CHANGELOG.md" ]]; then
            PREVIOUS_VERSION=$(grep -E "^## \[?v?[0-9]" CHANGELOG.md | head -n2 | tail -n1 | grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+')
        fi
        ;;
esac

if [[ -n "$PREVIOUS_VERSION" ]]; then
    # Clean up version format
    PREVIOUS_VERSION=$(echo "$PREVIOUS_VERSION" | sed 's/^v//')
    echo "✅ Previous version found: $PREVIOUS_VERSION"
    echo "$PREVIOUS_VERSION"
else
    echo "⚠️ Could not determine previous version using method: $METHOD"
    echo "Available methods: git-tags, bumpversion-commits, config-file, changelog"
    exit 1
fi