# Commit Message Conventions

This project uses **Conventional Commits** with **Gitmoji** for automated versioning and changelog generation.

## Format

```
<gitmoji> <type>(<optional-scope>): <description>

[optional body]

[optional footer(s)]
```

## Quick Reference

| Type | Gitmoji | Version Bump | Usage |
|------|---------|--------------|-------|
| `feat` | ✨ | Minor (1.0.0 → 1.1.0) | New features |
| `fix` | 🐛 | Patch (1.0.0 → 1.0.1) | Bug fixes |
| `perf` | ⚡ | Patch (1.0.0 → 1.0.1) | Performance improvements |
| `docs` | 📝 | None | Documentation only |
| `style` | 💄 | None | Code style (formatting, whitespace) |
| `refactor` | ♻️ | None | Code refactoring |
| `test` | ✅ | None | Adding or updating tests |
| `build` | 🏗️ | None | Build system or dependencies |
| `ci` | 👷 | None | CI/CD configuration |
| `chore` | 🔧 | None | Maintenance tasks |
| `revert` | ⏪ | None | Reverting previous commits |

## Breaking Changes

Add `!` after type or `BREAKING CHANGE:` in footer for major version bump (1.0.0 → 2.0.0):

```bash
# Option 1: Using !
git commit -m "✨ feat!: redesign model architecture"

# Option 2: Using footer
git commit -m "✨ feat: redesign model architecture

BREAKING CHANGE: Model checkpoint format has changed and is incompatible with v1.x"
```

## Examples

### ✅ Valid Commit Messages

```bash
# Feature with gitmoji
git commit -m "✨ feat: add support for Qwen2 model series"
git commit -m "✨ feat(tokenizer): add custom gene tokenization"

# Bug fix
git commit -m "🐛 fix: resolve megablocks ABI compatibility issue"
git commit -m "🐛 fix(data): handle missing gene tokens gracefully"

# Performance improvement
git commit -m "⚡ perf: optimize data loading by 30%"
git commit -m "⚡ perf(model): cache attention weights"

# Documentation
git commit -m "📝 docs: update installation instructions"
git commit -m "📝 docs(readme): add training tutorial link"

# Refactoring
git commit -m "♻️ refactor: simplify data collator logic"

# Build/Dependencies
git commit -m "🔧 chore: bump llm-foundry to 0.18.0"
git commit -m "⬆️ build(deps): upgrade PyTorch to 2.5.1"

# CI/CD
git commit -m "👷 ci: add conventional commits check"

# Tests
git commit -m "✅ test: add unit tests for tokenizer"

# Breaking change
git commit -m "✨ feat!: change model API to use new config format"

# Multi-line with body
git commit -m "✨ feat: add cell classification task

Implements a new task for classifying cell types based on gene expression.
Includes preprocessing, model architecture, and evaluation metrics."
```

### ❌ Invalid Commit Messages

```bash
# Missing type
git commit -m "✨ add new feature"              # ❌ No 'feat:'
git commit -m "add new feature"                 # ❌ No emoji or type

# Wrong case
git commit -m "✨ Feat: add new feature"        # ❌ Type should be lowercase
git commit -m "✨ FEAT: add new feature"        # ❌ Type should be lowercase

# Empty subject
git commit -m "✨ feat:"                        # ❌ Description is required

# Invalid type
git commit -m "✨ feature: add new capability"  # ❌ Use 'feat', not 'feature'
git commit -m "🐛 bugfix: fix issue"           # ❌ Use 'fix', not 'bugfix'
```

## Scopes (Optional)

Scopes provide additional context:

```bash
git commit -m "✨ feat(model): add attention mechanism"
git commit -m "🐛 fix(data): resolve dataloader bug"
git commit -m "📝 docs(api): update API reference"
git commit -m "⚡ perf(tokenizer): improve tokenization speed"
```

Common scopes:
- `model` - Model architecture changes
- `data` - Data loading and processing
- `tokenizer` - Tokenization logic
- `utils` - Utility functions
- `api` - Public API changes
- `cli` - Command-line interface
- `config` - Configuration
- `deps` - Dependencies

## Gitmoji Reference

You can use gitmoji with or without the shortcode:

```bash
# Using emoji directly
git commit -m "✨ feat: add feature"

# Using gitmoji shortcode
git commit -m ":sparkles: feat: add feature"
```

**Common gitmoji:**
- ✨ `:sparkles:` - New feature
- 🐛 `:bug:` - Bug fix
- 📝 `:memo:` - Documentation
- ⚡ `:zap:` - Performance
- 💄 `:lipstick:` - UI/Style
- ♻️ `:recycle:` - Refactoring
- ✅ `:white_check_mark:` - Tests
- 🔧 `:wrench:` - Configuration
- 🏗️ `:building_construction:` - Architecture
- 👷 `:construction_worker:` - CI/CD
- ⬆️ `:arrow_up:` - Upgrade dependencies
- ⬇️ `:arrow_down:` - Downgrade dependencies
- 🔒 `:lock:` - Security fixes
- 🎨 `:art:` - Code structure/format
- 🔥 `:fire:` - Remove code/files
- 🚀 `:rocket:` - Deploy/Release
- 🔖 `:bookmark:` - Version tag

Full list: https://gitmoji.dev/

## PR Title Conventions

PR titles must also follow conventional commits format. The PR title is used for squash merges to main.

**Good PR titles:**
- `✨ feat: add Qwen2 model support`
- `🐛 fix: resolve CUDA compatibility issue`
- `📝 docs: update README with examples`

**Bad PR titles:**
- `Add new feature` ❌
- `Fix bugs` ❌
- `Update docs` ❌

## CI Enforcement

The `.github/workflows/conventional-commits.yml` workflow checks:

1. **PR title** follows conventional commit format
2. **All commits** in PR follow conventional commit format

If checks fail, you'll see an error in the PR status checks with guidance on fixing the message.

## Tools

### Pre-commit Hook (Optional)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/compilerla/conventional-pre-commit
  rev: v3.0.0
  hooks:
    - id: conventional-pre-commit
      stages: [commit-msg]
```

### IDE Extensions

- **VSCode**: [Conventional Commits](https://marketplace.visualstudio.com/items?itemName=vivaxy.vscode-conventional-commits)
- **JetBrains**: [Conventional Commit](https://plugins.jetbrains.com/plugin/13389-conventional-commit)
- **Gitmoji**: [Gitmoji CLI](https://github.com/carloscuesta/gitmoji-cli)

### Command Line Helper

Use `gitmoji-cli` for interactive commit messages:

```bash
npm install -g gitmoji-cli

# Interactive commit
gitmoji -c
```

## Why Conventional Commits?

1. **Automated versioning**: Release Please determines version bumps from commit types
2. **Automated changelog**: Beautiful changelogs generated automatically
3. **Clear history**: Easy to understand what each commit does
4. **Better collaboration**: Team members understand changes at a glance
5. **Semantic versioning**: Proper semver based on change types

## References

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Gitmoji Guide](https://gitmoji.dev/)
- [Release Please Documentation](https://github.com/googleapis/release-please)
- [Commitlint](https://commitlint.js.org/)

## Need Help?

If you're unsure about a commit message:

1. Check examples above
2. Ask in PR comments
3. Use `chore:` for anything that doesn't fit other types
4. Remember: `feat:` for features, `fix:` for bugs, `docs:` for documentation
