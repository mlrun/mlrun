#!/bin/bash
# GitHub CLI-based release notes generator
# Requires: gh CLI tool

set -e

VERSION=${1:-}
PREVIOUS_VERSION=${2:-}
OUTPUT_FILE=${3:-"github_release_notes.md"}

if [[ -z "$VERSION" || -z "$PREVIOUS_VERSION" ]]; then
    echo "❌ Usage: $0 <version> <previous_version> [output_file]"
    echo "Example: $0 v1.13.205 v1.13.204 release_notes.md"
    exit 1
fi

echo "📝 Generating release notes with GitHub CLI..."

# Generate release notes using GitHub's API
gh api repos/:owner/:repo/releases/generate-notes \
  -f tag_name="$VERSION" \
  -f target_commitish="$(git rev-parse HEAD)" \
  -f previous_tag_name="$PREVIOUS_VERSION" \
  --jq '.body' > "$OUTPUT_FILE"

echo "✅ GitHub release notes generated: $OUTPUT_FILE"

# Add custom sections
cat >> "$OUTPUT_FILE" << EOF

## 📊 Release Statistics

$(git rev-list --count ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null || echo "N/A") commits since $PREVIOUS_VERSION

### 👥 Contributors
$(git shortlog -sn ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null | head -10)

### 🔗 Links
- **Full Changelog**: https://github.com/mlrun/mlrun/compare/${PREVIOUS_VERSION}...${VERSION}
- **Documentation**: https://docs.mlrun.org
- **Release Downloads**: https://github.com/mlrun/mlrun/releases/tag/${VERSION}

---
*Generated automatically by GitHub CLI*
EOF

echo "📈 Enhanced release notes ready!"