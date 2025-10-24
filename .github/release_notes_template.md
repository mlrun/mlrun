# Release Notes Template for MLRun

## 🎯 **Release Information**
- **Version**: {VERSION}
- **Release Date**: {DATE}
- **Previous Version**: {PREVIOUS_VERSION}
- **Release Type**: {RELEASE_TYPE} (Major/Minor/Patch)

## 🚀 **What's New**

### ✨ New Features
{FEATURES_LIST}

### 🐛 Bug Fixes  
{BUG_FIXES_LIST}

### 🔧 Improvements
{IMPROVEMENTS_LIST}

### ⚠️ Breaking Changes
{BREAKING_CHANGES_LIST}

### 📚 Documentation Updates
{DOCUMENTATION_LIST}

## 🔍 **Technical Details**

### 📊 Statistics
- **Total Commits**: {COMMIT_COUNT}
- **Contributors**: {CONTRIBUTOR_COUNT}
- **Files Changed**: {FILES_CHANGED}
- **Lines Added**: {LINES_ADDED}
- **Lines Removed**: {LINES_REMOVED}

### 🎯 **Highlights**
{HIGHLIGHTS}

## 📦 **Installation & Upgrade**

### Fresh Installation
```bash
pip install mlrun=={VERSION}
```

### Upgrade from Previous Version
```bash
pip install --upgrade mlrun=={VERSION}
```

### Docker Images
```bash
# MLRun API
docker pull ghcr.io/mlrun/mlrun-api:{VERSION}

# MLRun Runtime
docker pull ghcr.io/mlrun/mlrun:{VERSION}

# Jupyter with MLRun
docker pull ghcr.io/mlrun/jupyter:{VERSION}
```

## 🔗 **Links**
- **Full Changelog**: https://github.com/mlrun/mlrun/compare/{PREVIOUS_VERSION}...{VERSION}
- **Documentation**: https://docs.mlrun.org
- **Issues**: https://github.com/mlrun/mlrun/issues
- **Discussions**: https://github.com/mlrun/mlrun/discussions

## 🤝 **Contributors**
{CONTRIBUTORS_LIST}

## ⚠️ **Known Issues**
{KNOWN_ISSUES}

---
*This release was automatically generated on {GENERATION_DATE}*