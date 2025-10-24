#!/bin/bash
# Enhanced release notes generation script
# Usage: ./scripts/generate-release-notes.sh <version> <previous_version> [branch]

set -e

VERSION=${1:-}
PREVIOUS_VERSION=${2:-}
BRANCH=${3:-$(git branch --show-current)}
OUTPUT_FILE=${4:-"release_notes.md"}

if [[ -z "$VERSION" || -z "$PREVIOUS_VERSION" ]]; then
    echo "❌ Usage: $0 <version> <previous_version> [branch] [output_file]"
    echo "Example: $0 v1.13.205 v1.13.204 main release_notes.md"
    exit 1
fi

echo "📝 Generating release notes..."
echo "🎯 Version: $VERSION"
echo "🔙 Previous: $PREVIOUS_VERSION"
echo "🌿 Branch: $BRANCH"
echo "📄 Output: $OUTPUT_FILE"

# Use your existing Makefile target with enhanced output
MLRUN_VERSION="$VERSION" \
MLRUN_OLD_VERSION="$PREVIOUS_VERSION" \
MLRUN_RELEASE_BRANCH="$BRANCH" \
MLRUN_RELEASE_NOTES_OUTPUT_FILE="$OUTPUT_FILE" \
MLRUN_RAISE_ON_ERROR="false" \
MLRUN_SKIP_CLONE="true" \
make release-notes

echo "✅ Release notes generated: $OUTPUT_FILE"

# Additional enhancements
echo ""
echo "📊 Release Statistics:"
echo "---"

# Count commits between versions
COMMIT_COUNT=$(git rev-list --count ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null || echo "N/A")
echo "📝 Commits: $COMMIT_COUNT"

# Count contributors
CONTRIBUTORS=$(git shortlog -sn ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null | wc -l || echo "N/A")
echo "👥 Contributors: $CONTRIBUTORS"

# Get date range
START_DATE=$(git log -1 --format="%ai" $PREVIOUS_VERSION 2>/dev/null | cut -d' ' -f1 || echo "N/A")
END_DATE=$(git log -1 --format="%ai" $VERSION 2>/dev/null | cut -d' ' -f1 || echo "N/A")
echo "📅 Date Range: $START_DATE to $END_DATE"

# Lines of code changes
if git rev-parse $PREVIOUS_VERSION >/dev/null 2>&1 && git rev-parse $VERSION >/dev/null 2>&1; then
    ADDITIONS=$(git diff --shortstat ${PREVIOUS_VERSION}..${VERSION} | grep -oE '[0-9]+ insertions?' | grep -oE '[0-9]+' || echo "0")
    DELETIONS=$(git diff --shortstat ${PREVIOUS_VERSION}..${VERSION} | grep -oE '[0-9]+ deletions?' | grep -oE '[0-9]+' || echo "0")
    echo "📈 Lines Added: ${ADDITIONS:-0}"
    echo "📉 Lines Removed: ${DELETIONS:-0}"
fi

echo ""
echo "🔗 Quick Links:"
echo "- Compare: https://github.com/mlrun/mlrun/compare/${PREVIOUS_VERSION}...${VERSION}"
echo "- Releases: https://github.com/mlrun/mlrun/releases"
echo "- Issues: https://github.com/mlrun/mlrun/issues"