# GitHub Actions CI/CD Setup

This directory contains GitHub Actions workflows for comprehensive testing, code quality, and deployment automation.

## Workflows

### 🧪 test.yml - Main Test Suite
**Triggers:** Push/PR to main/develop branches

**Features:**
- Tests across Python versions: 3.9, 3.10, 3.11, 3.12
- Tests across scikit-learn versions: 1.0.2 through 1.5.2
- Category encoders compatibility testing on select configurations
- Code coverage reporting (uploaded to Codecov)
- Minimal dependency testing
- Performance benchmarks and edge cases

**Matrix Strategy:**
- Core tests run on all Python × scikit-learn combinations
- Category encoders tested on Python 3.9, 3.10, 3.11, 3.12 with recent sklearn versions
- Python 3.12 excludes older sklearn versions due to compatibility
- Performance benchmarking across Python 3.10, 3.11, 3.12 with latest sklearn versions

### 🚀 release.yml - Release Automation
**Triggers:** GitHub releases, manual workflow dispatch

**Features:**
- Pre-release testing across multiple configurations
- Automated PyPI publishing with trusted publishing
- TestPyPI publishing for manual releases
- Build artifact management
- GitHub release creation with assets

**Security:** Uses OpenID Connect for secure PyPI publishing (no API tokens needed)

### 🔍 code-quality.yml - Code Quality Checks
**Triggers:** Push/PR to main/develop branches

**Features:**
- Linting with Ruff (code style, imports, complexity)
- Type checking with MyPy
- Security scanning with Bandit
- Dependency vulnerability checking with Safety
- Documentation style checking with pydocstyle
- Cross-platform compatibility testing (Ubuntu, Windows, macOS)

### 🎯 category-encoders.yml - Encoder Compatibility
**Triggers:** Push/PR, weekly schedule

**Features:**
- Tests multiple category_encoders versions (2.5.1 through 2.6.3 + latest)
- Python 3.9, 3.10, 3.11, 3.12 compatibility testing
- High cardinality category testing
- Memory usage validation
- Unknown category handling
- Development version testing from Git
- Strategic exclusions for older versions with Python 3.12

### ⚡ performance.yml - Performance Monitoring
**Triggers:** Weekly schedule, manual dispatch

**Features:**
- Comprehensive benchmarking across Python versions (3.10, 3.11, 3.12)
- Multi scikit-learn version testing (1.4.4, 1.5.2 by default)
- Memory profiling and analysis
- Performance regression detection
- Artifact retention for 30-90 days
- Cross-version performance comparison
- Matrix testing with appropriate exclusions for compatibility

## Configuration Files

### 📦 dependabot.yml
- Automated dependency updates
- Grouped updates for related packages
- Weekly schedule for Python and GitHub Actions dependencies

### 🪝 .pre-commit-config.yaml (Updated)
- Enhanced with security scanning (Bandit)
- Comprehensive code quality checks
- Quick test execution for fast feedback

### ⚙️ pyproject.toml (Enhanced)
- Added testing dependencies (pytest-cov, pytest-xdist, pytest-benchmark)
- Security tools (bandit, safety)
- Coverage configuration with 80% minimum threshold
- Test markers for better organization

## Package Management

All workflows use **uv** for fast and reliable dependency management:

- **Faster installs**: uv is significantly faster than pip
- **Better dependency resolution**: More reliable lock file handling
- **Improved caching**: Built-in dependency caching with `setup-uv` action
- **Virtual environment management**: Automatic virtual environment handling

### uv Configuration
- Uses `uv.lock` for deterministic builds
- Leverages GitHub Actions caching for improved performance
- Supports both development and production dependency groups
- Handles both PyPI and Git-based dependencies

### Benefits
- **Speed**: Up to 10x faster than pip for complex dependency trees
- **Reliability**: Better conflict resolution and version management
- **Consistency**: Same dependency versions across all environments
- **Developer Experience**: Simplified local development setup
No secrets required! This setup uses:
- **Trusted Publishing** for PyPI (configure in PyPI settings)
- **GITHUB_TOKEN** for releases (automatically provided)

### 2. PyPI Trusted Publishing Setup
1. Go to PyPI > Account Settings > Publishing
2. Add GitHub publisher:
   - Owner: `hadi-gharibi`
   - Repository: `sklean`
   - Workflow: `release.yml`
   - Environment: `pypi`

### 3. Branch Protection Rules
Recommended settings for `main` branch:
- Require status checks: `test`, `lint`, `security`
- Require up-to-date branches
- Require linear history

### 4. Environment Setup
Create environments in GitHub repository settings:
- **pypi**: For production releases
- **testpypi**: For test releases

## Test Matrix Details

### Python Versions
- **3.9**: Minimum supported, extensive testing
- **3.10**: LTS version, full compatibility
- **3.11**: Current stable, primary development target
- **3.12**: Latest version, excludes older sklearn versions

### scikit-learn Versions
- **1.0.2**: Legacy support (Python < 3.12 only)
- **1.1.3**: Extended compatibility
- **1.2.2**: Stable branch
- **1.3.2**: Feature release with category encoders
- **1.4.4**: Recent stable with category encoders
- **1.5.2**: Latest stable with category encoders

### Category Encoders Testing
- **Versions**: 2.5.1, 2.6.0, 2.6.1, 2.6.2, 2.6.3, latest, git-dev
- **Encoders**: BinaryEncoder, TargetEncoder, CatBoostEncoder, OneHotEncoder
- **Edge Cases**: High cardinality, unknown categories, memory usage

## Performance Monitoring

### Benchmarking
- **Frequency**: Weekly automated runs
- **Metrics**: Execution time, memory usage, accuracy
- **Comparison**: Cross-version performance analysis
- **Artifacts**: Charts, profiling reports, comparison data

### Regression Detection
- Automatic performance baseline comparison
- Memory usage threshold monitoring
- Benchmark completion validation
- Historical trend analysis

## Artifact Management

### Test Results
- **Coverage reports**: XML format for Codecov integration
- **Test outputs**: Detailed logs for debugging
- **Performance data**: JSON reports for analysis

### Security Reports
- **Bandit**: JSON security vulnerability reports
- **Safety**: Dependency vulnerability scans
- **Retention**: 30 days for quick access

### Performance Artifacts
- **Benchmark results**: Charts and raw data (30 days)
- **Memory profiles**: Detailed memory usage (30 days)
- **Performance summaries**: Historical comparison (90 days)

## Troubleshooting

### Common Issues

1. **Test failures on specific sklearn versions**
   - Check compatibility matrix in test.yml
   - Review sklearn changelog for breaking changes
   - Update exclusion rules if needed

2. **Coverage below threshold**
   - Add tests for uncovered code
   - Update coverage configuration in pyproject.toml
   - Use `--cov-report=html` locally for detailed analysis

3. **Security scan failures**
   - Review Bandit report for false positives
   - Add exclusions to pyproject.toml if needed
   - Update vulnerable dependencies

4. **Performance regression**
   - Check performance.yml logs for specific metrics
   - Compare with historical artifacts
   - Profile locally using examples/profiler_demo.py

### Local Testing
Run the same checks locally with uv:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run full test suite
uv run pytest tests/ --cov=sklean --cov-report=html

# Security scanning
uv run bandit -r src/
uv run safety check

# Performance benchmarking
cd examples && uv run python comprehensive_benchmark.py
```

## Monitoring & Alerts

### GitHub Actions
- **Failed workflow notifications**: Enable in repository settings
- **Status badges**: Add to README.md for visibility
- **Branch protection**: Prevent merges with failing tests

### Third-party Integrations
- **Codecov**: Coverage reporting and PR comments
- **Dependabot**: Automated dependency update PRs
- **Performance tracking**: Historical data in artifacts

## Future Enhancements

### Planned Additions
- **Docker testing**: Multi-architecture container testing
- **Documentation**: Automated API docs generation
- **Release notes**: Automatic changelog generation
- **Slack notifications**: Team alerts for critical failures
- **Performance dashboard**: Web interface for trend analysis

### Scaling Considerations
- **Matrix optimization**: Reduce redundant test combinations
- **Parallel execution**: Optimize workflow concurrency
- **Artifact storage**: Implement cleanup policies
- **Cost optimization**: Monitor GitHub Actions usage

---

For questions or issues with the CI/CD setup, please open an issue or contact the maintainers.
