# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.11.0]

### Added
- GitHub Actions cache implementation for Docker builds
- Automated changelog generation system
- Improved build workflow with proper commit SHA handling

### Changed
- Replaced registry-based Docker caching with GHA cache type
- Simplified GitHub Actions workflow logic
- Updated Makefile to remove obsolete cache pull commands

### Removed
- Obsolete Docker pull infrastructure for caching
- Legacy cache pull variables and commands

### Fixed
- Fixed comma parsing issues in Makefile cache variables
- Corrected GitHub SHA handling for PR triggers

---

## How to Update This Changelog

This changelog is automatically updated by our release process, but you can also update it manually:

### For Contributors:
- Use [Conventional Commits](https://www.conventionalcommits.org/) format in your commit messages
- Examples:
  - `feat: add new caching system`
  - `fix: resolve Docker build issues`
  - `docs: update installation guide`
  - `chore: update dependencies`

### Commit Types:
- `feat`: New features → **Added** section
- `fix`: Bug fixes → **Fixed** section  
- `docs`: Documentation → **Changed** section
- `style`: Code style changes → **Changed** section
- `refactor`: Code refactoring → **Changed** section
- `test`: Adding tests → **Added** section
- `chore`: Maintenance → **Changed** section
- `BREAKING CHANGE`: Breaking changes → **Breaking Changes** section

### Manual Updates:
If you need to add entries manually, follow this format:

```markdown
## [Version] - YYYY-MM-DD

### Added
- New features and capabilities

### Changed  
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Features removed in this version

### Fixed
- Bug fixes

### Security
- Security improvements
```