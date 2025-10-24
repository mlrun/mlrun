#!/bin/bash
# Bumpversion wrapper that tracks previous version
# Usage: ./scripts/bump-version.sh [patch|minor|major|rc]

set -e

BUMP_TYPE=${1:-patch}
OUTPUT_FILE=${2:-.version_history}

echo "🔧 Bumping version ($BUMP_TYPE)..."

# Capture current version before bumping
PREVIOUS_VERSION=$(grep "current_version" .bumpversion.cfg | cut -d'=' -f2 | xargs)
echo "📋 Current version: $PREVIOUS_VERSION"

# Perform the bump
echo "⬆️ Executing bumpversion $BUMP_TYPE..."
bumpversion "$BUMP_TYPE"

# Get new version after bumping
NEW_VERSION=$(grep "current_version" .bumpversion.cfg | cut -d'=' -f2 | xargs)
echo "🎯 New version: $NEW_VERSION"

# Store version history
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TIMESTAMP | $PREVIOUS_VERSION -> $NEW_VERSION | $BUMP_TYPE" >> "$OUTPUT_FILE"

# Also store in separate files for easy access
echo "$PREVIOUS_VERSION" > .previous_version
echo "$NEW_VERSION" > .current_version

# Export for use in CI/CD
cat > .version_info << EOF
PREVIOUS_VERSION=$PREVIOUS_VERSION
CURRENT_VERSION=$NEW_VERSION
BUMP_TYPE=$BUMP_TYPE
TIMESTAMP=$TIMESTAMP
EOF

echo "✅ Version bumped successfully!"
echo "📁 Version info saved to: $OUTPUT_FILE, .previous_version, .current_version, .version_info"

# Optionally show git status
echo ""
echo "📊 Git status after bump:"
git status --porcelain