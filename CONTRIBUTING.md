# Contributing to Stripje

Thank you for your interest in contributing to Stripje! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Getting Started

Before you begin contributing, please:

1. Read our [README.md](README.md) to understand the project
2. Check existing [issues](issues/) to see if your idea or bug has already been reported
3. Look at the [project structure](#project-structure) to understand the codebase

## Development Setup

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/Stripje.git
   cd Stripje
   ```

2. **Set up the development environment:**
   ```bash
   # Install dependencies using uv
   uv sync

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Verify the installation:**
   ```bash
   # Run tests to ensure everything is working
   uv run pytest tests/

   # Run a simple example
   uv run python examples/simple_example.py
   ```

### Project Structure

```
Stripje/
├── src/Stripje/           # Main package source code
│   ├── estimators/      # ML estimators
│   ├── transformers/    # Data transformers
│   ├── fast_pipeline.py # Core pipeline implementation
│   ├── profiling.py     # Performance profiling tools
│   ├── registry.py      # Component registry
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── examples/            # Usage examples
├── issues/              # Project issues and tasks
└── docs/                # Documentation (if applicable)
```

## Code Style Guidelines

### Python Code Style

- **PEP 8 Compliance:** Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- **Line Length:** Maximum 88 characters (Black formatter default)
- **Import Organization:** Use `isort` for import sorting
- **Type Hints:** Add type hints for all public functions and methods
- **Docstrings:** Use Google-style docstrings

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code with Black
uv run black src/ tests/ examples/

# Sort imports with isort
uv run isort src/ tests/ examples/

# Lint with flake8
uv run flake8 src/ tests/ examples/

# Type checking with mypy
uv run mypy src/
```

### Naming Conventions

- **Classes:** Use `PascalCase` (e.g., `FastPipeline`)
- **Functions/Variables:** Use `snake_case` (e.g., `fit_transform`)
- **Constants:** Use `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private Members:** Prefix with underscore (e.g., `_internal_method`)

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, concise commit messages
   - Include tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Before submitting:**
   ```bash
   # Run the full test suite
   uv run pytest tests/

   # Check code style
   uv run black --check src/ tests/ examples/
   uv run isort --check-only src/ tests/ examples/
   uv run flake8 src/ tests/ examples/

   # Run type checking
   uv run mypy src/
   ```

4. **Submit your pull request:**
   - Use a clear, descriptive title
   - Reference any related issues (e.g., "Fixes #123")
   - Provide a detailed description of changes
   - Include screenshots for UI changes (if applicable)

5. **Code Review Process:**
   - Maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your PR will be merged

## Issue Reporting Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Try the latest version** to see if the issue persists

### Creating a Good Issue

Use our issue templates when available. Include:

#### For Bug Reports:
- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment details** (Python version, OS, package versions)
- **Code example** demonstrating the issue
- **Error messages** and stack traces

#### For Feature Requests:
- **Clear description** of the proposed feature
- **Use case** explaining why it's needed
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

#### For Performance Issues:
- **Benchmark results** showing the performance problem
- **Profiling data** if available
- **Dataset characteristics** (size, types, etc.)
- **Hardware specifications**

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_fast_pipeline.py

# Run with coverage
uv run pytest tests/ --cov=src/Stripje --cov-report=html

# Run performance benchmarks
uv run python examples/benchmark.py
```

### Writing Tests

- **Test Coverage:** Aim for >90% test coverage
- **Test Types:** Include unit tests, integration tests, and performance tests
- **Naming:** Use descriptive test names (e.g., `test_pipeline_handles_empty_dataset`)
- **Fixtures:** Use pytest fixtures for common test data
- **Assertions:** Use clear, specific assertions

### Test Structure

```python
def test_feature_description():
    """Test that feature works as expected."""
    # Arrange
    input_data = create_test_data()
    expected_result = calculate_expected()

    # Act
    actual_result = function_under_test(input_data)

    # Assert
    assert actual_result == expected_result
```

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def transform_data(data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Transform input data using specified method.

    Args:
        data: Input DataFrame to transform.
        method: Transformation method to use. Options: "standard", "minmax".

    Returns:
        Transformed DataFrame with same shape as input.

    Raises:
        ValueError: If method is not supported.

    Example:
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> transformed = transform_data(df, method="standard")
        >>> print(transformed.shape)
        (3, 1)
    """
```

### Code Comments

- **When to comment:** Explain why, not what
- **Complex algorithms:** Add explanatory comments
- **Performance optimizations:** Document reasoning
- **TODOs:** Include issue numbers when possible

## Getting Help

- **Questions:** Open a discussion or issue
- **Chat:** Join our community channels (if available)
- **Documentation:** Check the README and code examples

## Recognition

Contributors will be acknowledged in:
- Release notes for significant contributions
- README.md contributors section
- Git commit history

Thank you for contributing to Stripje! 🚀
