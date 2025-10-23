# Automated Release Process with Release Please

This repository uses **Release Please** by Google to automate versioning, changelog generation, and releases. Release Please creates a "Release PR" that tracks all changes since the last release. When you merge it, the release is automatically published to PyPI.

## How It Works

### The Release PR Workflow

1. **You push to main** (via PR merge or direct commit)
2. **Release Please analyzes commits** since last release
3. **Automatic Release PR created/updated** containing:
   - Version bump in `tahoex/_version.py`
   - Auto-generated `CHANGELOG.md` with all changes
   - GitHub Release notes
4. **You merge the Release PR** when ready
5. **Automatic publishing**:
   - Creates Git tag
   - Creates GitHub Release
   - Builds package
   - Publishes to PyPI

### Commit Message Format

Release Please uses **Conventional Commits** to determine version bumps:

```bash
# Feature (minor bump: 1.0.0 → 1.1.0)
git commit -m "feat: add support for Qwen2 models"
git commit -m "✨ feat: add cell classification task"

# Bug fix (patch bump: 1.0.0 → 1.0.1)
git commit -m "fix: resolve megablocks ABI compatibility"
git commit -m "🐛 fix: handle missing gene tokens"

# Performance (patch bump: 1.0.0 → 1.0.1)
git commit -m "perf: optimize data loading by 30%"
git commit -m "⚡ perf: cache tokenizer lookups"

# Breaking change (major bump: 1.0.0 → 2.0.0)
git commit -m "feat!: redesign model architecture"
# OR include in body:
git commit -m "feat: redesign model architecture

BREAKING CHANGE: Model checkpoint format has changed"

# Non-release changes (no version bump)
git commit -m "docs: update README"
git commit -m "📝 docs: add training tutorial"
git commit -m "chore: update pre-commit hooks"
git commit -m "🔧 chore: bump dependencies"
```

### Gitmoji with Conventional Commits

You can use both! Put gitmoji before the conventional commit type:

```bash
git commit -m "✨ feat: add new feature"
git commit -m "🐛 fix: resolve bug"
git commit -m "📝 docs: update docs"
git commit -m "♻️ refactor: simplify code"
git commit -m "⚡ perf: improve performance"
git commit -m "🔧 chore: update config"
```

**Important**: The conventional commit type (`feat:`, `fix:`, etc.) must be present for Release Please to recognize it.

## Release Process

### Automatic Process (Recommended)

1. **Develop and commit** using conventional commit messages:
   ```bash
   git commit -m "✨ feat: add support for new model"
   git push
   ```

2. **Release Please creates/updates PR** automatically:
   - PR title: `chore(main): release 1.1.0`
   - Contains version bump and changelog
   - Updates with each new commit to main

3. **Review the Release PR**:
   - Check version number is correct
   - Review auto-generated changelog
   - Verify all changes are included

4. **Merge the Release PR**:
   - Click "Merge" when ready to release
   - Automatic actions triggered:
     - Git tag created (e.g., `v1.1.0`)
     - GitHub Release published
     - Package built and published to PyPI

5. **Done!** Check:
   - PyPI: https://pypi.org/project/tahoex/
   - GitHub Releases: https://github.com/tahoebio/tahoe-x1/releases

### Manual Version Control (If Needed)

If you need to control the exact version:

1. **Edit Release PR** before merging:
   - Change version in `tahoex/_version.py`
   - Update `CHANGELOG.md` if needed
   - Commit to the release PR branch

2. **Force specific version bump** using commit message:
   ```bash
   # Force major bump
   git commit -m "feat!: breaking change"

   # Force minor bump
   git commit -m "feat: new feature"

   # Force patch bump
   git commit -m "fix: bug fix"
   ```

## Version Bumping Rules

Release Please uses these rules:

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `feat:` | Minor (1.0.0 → 1.1.0) | New features |
| `fix:` | Patch (1.0.0 → 1.0.1) | Bug fixes |
| `perf:` | Patch (1.0.0 → 1.0.1) | Performance improvements |
| `feat!:` or `BREAKING CHANGE:` | Major (1.0.0 → 2.0.0) | Breaking changes |
| `docs:`, `chore:`, `style:`, etc. | None | Non-release changes |

**Multiple commits**: Largest bump wins (breaking > feat > fix)

## Configuration

### Release Please Config (`.release-please-config.json`)

Configures changelog sections with emojis:
- ✨ Features (`feat:`)
- 🐛 Bug Fixes (`fix:`)
- ⚡ Performance Improvements (`perf:`)
- 📝 Documentation (`docs:`)
- ♻️ Code Refactoring (`refactor:`)
- 🏗️ Build System (`build:`)

Hidden from changelog: `style:`, `test:`, `ci:`, `chore:`

### Version Manifest (`.release-please-manifest.json`)

Tracks current version: `{"." : "1.0.4"}`

Updated automatically by Release Please.

## Setup Requirements

### PyPI Trusted Publishing

Set up once at https://pypi.org/manage/account/publishing/:

- **PyPI Project Name**: `tahoex`
- **Owner**: `tahoebio`
- **Repository name**: `tahoe-x1`
- **Workflow name**: `release-please.yml`
- **Environment name**: `pypi`

### GitHub Permissions

The workflow needs:
- `contents: write` - Create releases and tags
- `pull-requests: write` - Create/update Release PRs
- `id-token: write` - PyPI trusted publishing

These are configured in `.github/workflows/release-please.yml`.

## Examples

### Example Release PR

```markdown
## [1.1.0](https://github.com/tahoebio/tahoe-x1/compare/v1.0.4...v1.1.0) (2025-10-22)

### ✨ Features

* add support for Qwen2 model series ([#123](https://github.com/tahoebio/tahoe-x1/pull/123))
* implement cell classification task ([abc1234](https://github.com/tahoebio/tahoe-x1/commit/abc1234))

### 🐛 Bug Fixes

* resolve megablocks ABI compatibility issue ([def5678](https://github.com/tahoebio/tahoe-x1/commit/def5678))
* handle missing gene tokens gracefully ([#124](https://github.com/tahoebio/tahoe-x1/pull/124))

### ⚡ Performance Improvements

* optimize data loading by 30% ([ghi9012](https://github.com/tahoebio/tahoe-x1/commit/ghi9012))
```

### Example Commit History

```bash
✨ feat: add Qwen2 model support
🐛 fix: resolve CUDA compatibility
📝 docs: update README installation
⚡ perf: cache tokenizer lookups
🔧 chore: update pre-commit hooks
✨ feat: add cell classification
🐛 fix: handle edge case in data loading
```

**Result**: Release PR created for version 1.1.0 with all `feat:` and `fix:` commits in changelog.

## Troubleshooting

### Release PR Not Created

**Check commit messages**: Must use conventional commit format (`feat:`, `fix:`, etc.)

```bash
# Won't trigger release
git commit -m "add new feature"
git commit -m "✨ add new feature"

# Will trigger release
git commit -m "feat: add new feature"
git commit -m "✨ feat: add new feature"
```

### Wrong Version Number

**Edit the Release PR**:
1. Go to the Release PR
2. Edit `tahoex/_version.py` directly in the PR
3. Commit changes to the PR branch
4. Merge when correct

**Or force correct bump** with commit type:
- Want minor? Use `feat:`
- Want patch? Use `fix:`
- Want major? Use `feat!:` or `BREAKING CHANGE:`

### Release PR Keeps Updating

This is normal! Release Please updates the PR with each new commit to `main` until you merge it.

To release:
1. Stop merging PRs to main temporarily
2. Merge the Release PR
3. Resume normal development

### PyPI Publishing Failed

**Check trusted publishing setup**: Verify PyPI publisher settings match exactly.

**Check workflow logs**: Go to Actions tab → Release Please workflow → publish-to-pypi job

**Manual publish** if needed:
```bash
git checkout v1.1.0  # the tag created by Release Please
python -m build
twine upload dist/*
```

### Need to Release Without Conventional Commits

You can manually create a Release PR:
1. Create branch from main
2. Edit `tahoex/_version.py`
3. Update `CHANGELOG.md`
4. Create PR with title: `chore(main): release X.Y.Z`
5. Merge to trigger release

## Best Practices

1. **Use conventional commits** - Required for automation
2. **Combine gitmoji and conventional** - `✨ feat:` for the best of both worlds
3. **Let Release PR accumulate** - Don't merge immediately, let it collect multiple changes
4. **Review before merging** - Check changelog makes sense
5. **Keep PR descriptions clean** - They appear in release notes
6. **Use breaking change carefully** - Major version bumps signal users

## Comparison to Other Approaches

| Approach | Version Control | Commit Format | Automation |
|----------|----------------|---------------|------------|
| **Release Please** (this repo) | Automatic | Conventional Commits required | Full |
| Tag-based (llm-foundry) | Manual | Any format | Partial |
| Semantic Release | Automatic | Conventional Commits required | Full |

**Why Release Please?**
- Best of both worlds: manual control via PR review, automatic execution
- Clear changelog automatically generated
- Works great with gitmoji + conventional commits
- Used by major projects (Google, Anthropic, etc.)

## Additional Resources

- [Release Please Documentation](https://github.com/googleapis/release-please)
- [Conventional Commits Spec](https://www.conventionalcommits.org/)
- [Gitmoji Guide](https://gitmoji.dev/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Semantic Versioning](https://semver.org/)

## Migration Note

Current version: `1.0.4` (tracked in `.release-please-manifest.json`)

On next conventional commit to main, Release Please will create the first Release PR with version `1.0.5`, `1.1.0`, or `2.0.0` depending on commit type.
