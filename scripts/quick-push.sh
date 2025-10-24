#!/bin/bash
# Quick push script for maintaining single commit on feature branch
# Usage: ./scripts/quick-push.sh [optional-commit-message]

set -e

BRANCH_NAME=$(git branch --show-current)
COMMIT_MESSAGE=${1:-"ci(core): Release procedure implementation"}

echo "🔄 Quick push for single-commit workflow"
echo "📋 Branch: $BRANCH_NAME"
echo "💬 Message: $COMMIT_MESSAGE"

# Check if we have changes
if [[ -z $(git status --porcelain) ]]; then
    echo "⚠️ No changes to commit"
    exit 0
fi

# Add all changes
echo "➕ Staging changes..."
git add .

# Check if this is the first commit or if we should amend
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "🔧 Amending existing commit..."
    git commit --amend -m "$COMMIT_MESSAGE"
else
    echo "🆕 Creating first commit..."
    git commit -m "$COMMIT_MESSAGE"
fi

# Push with force-with-lease for safety
echo "🚀 Force pushing with lease..."
git push --force-with-lease origin "$BRANCH_NAME"

echo "✅ Successfully pushed single commit to $BRANCH_NAME!"

# Show current status
echo ""
echo "📊 Current status:"
git log --oneline -1
git status --porcelain
