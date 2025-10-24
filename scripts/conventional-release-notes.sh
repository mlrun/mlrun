#!/bin/bash
# Conventional commits-based release notes generator
# Parses commit messages following conventional commit format

set -e

VERSION=${1:-}
PREVIOUS_VERSION=${2:-}
OUTPUT_FILE=${3:-"conventional_release_notes.md"}

if [[ -z "$VERSION" || -z "$PREVIOUS_VERSION" ]]; then
    echo "❌ Usage: $0 <version> <previous_version> [output_file]"
    exit 1
fi

echo "📝 Generating conventional commit-based release notes..."

# Create temporary files for different sections
TEMP_DIR=$(mktemp -d)
FEATURES="$TEMP_DIR/features.txt"
FIXES="$TEMP_DIR/fixes.txt"
BREAKING="$TEMP_DIR/breaking.txt"
OTHERS="$TEMP_DIR/others.txt"

# Get commits between versions
git log --oneline "${PREVIOUS_VERSION}..${VERSION}" | while read -r line; do
    commit_hash=$(echo "$line" | cut -d' ' -f1)
    commit_msg=$(echo "$line" | cut -d' ' -f2-)
    
    case "$commit_msg" in
        feat*|feature*)
            echo "- $commit_msg ($commit_hash)" >> "$FEATURES"
            ;;
        fix*|bug*)
            echo "- $commit_msg ($commit_hash)" >> "$FIXES"
            ;;
        *BREAKING*|*breaking*)
            echo "- $commit_msg ($commit_hash)" >> "$BREAKING"
            ;;
        *)
            echo "- $commit_msg ($commit_hash)" >> "$OTHERS"
            ;;
    esac
done

# Generate release notes
cat > "$OUTPUT_FILE" << EOF
# Release Notes - $VERSION

## 🚀 Features
EOF

if [[ -f "$FEATURES" ]]; then
    cat "$FEATURES" >> "$OUTPUT_FILE"
else
    echo "No new features in this release." >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << EOF

## 🐛 Bug Fixes
EOF

if [[ -f "$FIXES" ]]; then
    cat "$FIXES" >> "$OUTPUT_FILE"
else
    echo "No bug fixes in this release." >> "$OUTPUT_FILE"
fi

if [[ -f "$BREAKING" && -s "$BREAKING" ]]; then
    cat >> "$OUTPUT_FILE" << EOF

## ⚠️ BREAKING CHANGES
EOF
    cat "$BREAKING" >> "$OUTPUT_FILE"
fi

if [[ -f "$OTHERS" && -s "$OTHERS" ]]; then
    cat >> "$OUTPUT_FILE" << EOF

## 🔧 Other Changes
EOF
    cat "$OTHERS" >> "$OUTPUT_FILE"
fi

# Add footer
cat >> "$OUTPUT_FILE" << EOF

## 📊 Statistics
- **Commits**: $(git rev-list --count ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null || echo "N/A")
- **Contributors**: $(git shortlog -sn ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null | wc -l || echo "N/A")
- **Files Changed**: $(git diff --name-only ${PREVIOUS_VERSION}..${VERSION} 2>/dev/null | wc -l || echo "N/A")

**Full Changelog**: https://github.com/mlrun/mlrun/compare/${PREVIOUS_VERSION}...${VERSION}
EOF

# Cleanup
rm -rf "$TEMP_DIR"

echo "✅ Conventional release notes generated: $OUTPUT_FILE"