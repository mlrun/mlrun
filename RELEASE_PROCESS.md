# Release Process

This document describes the release process for MLRun.

## Overview

Our release process is designed to be:
- **Automated** - Minimal manual intervention
- **Consistent** - Same process every time  
- **Transparent** - Clear changelog and versioning
- **Safe** - Built-in safeguards and rollback capabilities

## Release Types

### 1. Patch Release (1.2.3 → 1.2.4)
- **When**: Bug fixes, security patches, minor improvements
- **Branch**: `release/1.2.x` or `development`
- **Frequency**: As needed

### 2. Minor Release (1.2.3 → 1.3.0) 
- **When**: New features, non-breaking changes
- **Branch**: `development` → `release/1.3.x`
- **Frequency**: Monthly/Quarterly

### 3. Major Release (1.2.3 → 2.0.0)
- **When**: Breaking changes, major rewrites
- **Branch**: `development` → `release/2.0.x` 
- **Frequency**: Yearly/As needed

### 4. Pre-release (1.2.3 → 1.3.0-rc.1)
- **When**: Testing before major/minor releases
- **Branch**: `release/1.3.x`
- **Frequency**: Before minor/major releases

## Quick Release Guide

### Step 1: Prepare the Release
```bash
# 1. Create release branch (for minor/major releases)
git checkout development
git pull origin development
git checkout -b release/v1.3.x

# 2. Update version and changelog
./scripts/prepare-release.sh --type minor  # or patch, major, prerelease

# 3. Test the release candidate
make test
make docker-images
```

### Step 2: Create the Release
```bash
# 1. Tag the release
git tag -a v1.3.0 -m "Release v1.3.0"

# 2. Push the tag (triggers release workflow)
git push origin v1.3.0

# 3. Push release branch
git push origin release/v1.3.x
```

### Step 3: Publish
- GitHub Actions automatically builds and pushes Docker images
- GitHub Release is created with changelog
- PyPI package is published

## Changelog Management

### Conventional Commits
We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation:

```bash
feat: add new caching system          → Added section
fix: resolve Docker build issues      → Fixed section  
docs: update installation guide       → Changed section
chore: update dependencies            → Changed section
BREAKING CHANGE: remove old API       → Breaking Changes section
```

### Manual Changelog Updates
You can also update the changelog manually:

1. Edit `CHANGELOG.md`
2. Add entries under `## [Unreleased]` section
3. Follow the format:
   ```markdown
   ## [Unreleased]
   
   ### Added
   - New features
   
   ### Changed
   - Changes in existing functionality
   
   ### Fixed
   - Bug fixes
   ```

## Branch Strategy

```
development (main development)
    ├── feature/new-caching-system
    ├── fix/docker-build-issue
    └── release/v1.3.x (release preparation)
            └── hotfix/critical-fix → release/v1.3.x → v1.3.1
```

### Development Flow
1. **Feature branches** → `development` (via PR)
2. **Release branches** created from `development` 
3. **Hotfixes** can target release branches directly
4. **Tags** created on release branches

## Automated Release Workflow

### Triggering a Release
```bash
# Option 1: GitHub UI
# Go to Actions → Release → Run workflow
# Select release type (patch/minor/major/prerelease)

# Option 2: CLI (with gh CLI)
gh workflow run release.yml -f release_type=minor

# Option 3: Manual tag (legacy)
git tag v1.3.0 && git push origin v1.3.0
```

### What Happens Automatically
1. **Version Calculation** - Automatically bumps based on type
2. **Changelog Generation** - From conventional commits  
3. **Docker Build** - All images with new version
4. **GitHub Release** - Created with changelog and assets
5. **PyPI Publish** - Python package uploaded
6. **Notifications** - Slack/email notifications (if configured)

## Version Numbering

We follow [Semantic Versioning (SemVer)](https://semver.org/):

- **MAJOR**: Breaking changes (2.0.0)
- **MINOR**: New features, backwards compatible (1.3.0)  
- **PATCH**: Bug fixes, backwards compatible (1.2.4)
- **PRERELEASE**: Pre-release versions (1.3.0-rc.1)

### Examples
- `1.2.3` - Stable release
- `1.3.0-rc.1` - Release candidate  
- `1.2.4-hotfix.1` - Hotfix pre-release
- `2.0.0-beta.1` - Beta release

## Docker Image Versioning

All Docker images are tagged with:
- **Version tag**: `v1.3.0`
- **Latest tag**: `latest` (for stable releases)
- **Branch tag**: `development`, `release-1.3.x` (for development)

```bash
# Examples
ghcr.io/mlrun/mlrun:v1.3.0
ghcr.io/mlrun/mlrun:latest  
ghcr.io/mlrun/mlrun:development
ghcr.io/mlrun/mlrun-api:v1.3.0
ghcr.io/mlrun/jupyter:v1.3.0
```

## Rollback Process

### If Release is Problematic
```bash
# 1. Delete the problematic tag
git tag -d v1.3.0
git push origin :refs/tags/v1.3.0

# 2. Delete GitHub release (via UI or CLI)
gh release delete v1.3.0

# 3. Fix issues and re-release
git tag v1.3.1
git push origin v1.3.1
```

### Emergency Hotfix
```bash
# 1. Create hotfix branch from last stable tag
git checkout v1.2.3
git checkout -b hotfix/critical-security-fix

# 2. Make fixes and test
# ... make changes ...
make test

# 3. Create patch release
git tag v1.2.4
git push origin v1.2.4
```

## Release Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Breaking changes documented  
- [ ] Migration guide ready (if needed)
- [ ] Security review completed

### Release
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Tag created and pushed
- [ ] Docker images built and pushed
- [ ] GitHub release created
- [ ] PyPI package published

### Post-Release  
- [ ] Release notes published
- [ ] Documentation site updated
- [ ] Community notifications sent
- [ ] Monitoring release metrics
- [ ] Close milestone/project

## Troubleshooting

### Common Issues

**"Version already exists"**
```bash
# Delete existing tag and recreate
git tag -d v1.3.0
git push origin :refs/tags/v1.3.0
git tag v1.3.0
git push origin v1.3.0
```

**"Docker build fails"**
```bash
# Check build logs in GitHub Actions
# Try building locally first:
make docker-images MLRUN_VERSION=1.3.0
```

**"Changelog is empty"** 
```bash
# Generate manually or add entries:
conventional-changelog -p angular -i CHANGELOG.md -s
```

## Scripts and Tools

### `scripts/prepare-release.sh`
```bash
./scripts/prepare-release.sh --type patch   # 1.2.3 → 1.2.4
./scripts/prepare-release.sh --type minor   # 1.2.3 → 1.3.0  
./scripts/prepare-release.sh --type major   # 1.2.3 → 2.0.0
./scripts/prepare-release.sh --version 1.5.0  # Custom version
```

### `scripts/generate-changelog.sh`
```bash
./scripts/generate-changelog.sh             # Generate from last tag
./scripts/generate-changelog.sh v1.2.0      # Generate from specific tag
```

This process ensures consistent, reliable, and automated releases! 🚀