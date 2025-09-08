# Stripje - Make sklearn pipelines lean

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-ye## 🛠️ Development

### Setup Development Environment

1. Clone the repository:
```bash
git clone https://github.com/hadi-gharibi/stripje.git
cd stripje
```

2. Install all dependencies (including optional ones for full testing):
```bash
uv sync --all-extras
```

3. Install pre-commit hooks:
```bash
uv run pre-commit install
```

### Code Quality Tools

This project uses modern Python development tools:

- **Ruff**: Fast linting, formatting, and import sorting (replaces black + flake8 + isort)
- **MyPy**: Static type checking
- **pre-commit**: Automated code quality checks

Run code quality checks:

```bash
# Lint and auto-fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Testing

Run tests with all dependencies (recommended for development):

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=stripje
```https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/hadi-gharibi/stripje)

A high-performance single-row inference compiler for scikit-learn pipelines. Make your sklearn pipelines lean and efficient by converting trained pipelines into optimized Python functions that avoid numpy overhead for single-row predictions.

## 🚀 Quick Start

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stripje import compile_pipeline

# Create and fit your pipeline as usual
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Compile the pipeline for fast single-row inference
fast_predict = compile_pipeline(pipeline)

# Use the compiled function for single-row predictions
test_row = [1.2, -0.5, 0.8, -1.1]
prediction = fast_predict(test_row)
```

## 📦 Installation

### Using pip

```bash
pip install stripje
```

### Using uv (recommended)

```bash
uv add stripje
```

### Development Installation

```bash
git clone https://github.com/hadi-gharibi/stripje.git
cd stripje
uv install
```

## 🎯 Problem & Solution

**Problem:** Standard scikit-learn pipelines are optimized for batch processing. When making predictions on single rows, the numpy operations introduce significant overhead, making single-row inference slower than necessary.

**Solution:** The Fast Pipeline Compiler analyzes a trained pipeline and creates a new function that:
- Extracts all necessary parameters from fitted transformers/estimators
- Creates optimized single-row functions for each step
- Composes them into a single fast prediction function

## 🔧 Supported Components

### Preprocessing Transformers
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `MaxAbsScaler`
- `Normalizer`
- `OneHotEncoder`
- `OrdinalEncoder`
- `LabelEncoder`
- `QuantileTransformer`
- `SelectKBest`

### Estimators
- `LogisticRegression`
- `LinearRegression`
- `RandomForestClassifier`
- `DecisionTreeClassifier`
- `GaussianNB`

### Composite
- `ColumnTransformer` (recursively compiles sub-transformers)

## 📖 Usage Examples

### Basic Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stripje import compile_pipeline

# Create and fit your pipeline as usual
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Compile the pipeline for fast single-row inference
fast_predict = compile_pipeline(pipeline)

# Use the compiled function for single-row predictions
test_row = [1.2, -0.5, 0.8, -1.1]
prediction = fast_predict(test_row)
```

### ColumnTransformer Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Create a pipeline with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['category', 'region'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit and compile
pipeline.fit(X_train, y_train)
fast_predict = compile_pipeline(pipeline)

# Single-row prediction
row = [25, 50000, 'A', 'North']  # [age, income, category, region]
prediction = fast_predict(row)
```

## ⚡ Performance

Typical speedups range from 2-10x for single-row inference, depending on the pipeline complexity. The more transformers in your pipeline, the greater the speedup.

## 📚 API Reference

### `compile_pipeline(pipeline)`

Compiles a fitted scikit-learn pipeline into a fast single-row prediction function.

**Parameters:**
- `pipeline`: A fitted scikit-learn Pipeline object

**Returns:**
- A function that takes a single row (list/array) and returns the prediction

**Raises:**
- `ValueError`: If any step in the pipeline is not supported

### `get_supported_transformers()`

Returns a list of supported transformer/estimator types.

**Returns:**
- List of supported classes

## 🔌 Extending Support

To add support for a new transformer:

```python
from stripje import register_step_handler

@register_step_handler(YourTransformer)
def handle_your_transformer(step):
    # Extract parameters from the fitted step
    param1 = step.param1_
    param2 = step.param2_

    def transform_one(x):
        # Implement single-row transformation logic
        result = []
        for val in x:
            # Your transformation logic here
            transformed_val = val * param1 + param2
            result.append(transformed_val)
        return result

    return transform_one
```

## ⚠️ Limitations

- Only supports the listed transformers/estimators
- Input must be provided as a list or array-like structure
- No support for sparse matrices
- Some complex transformers may have approximations (e.g., QuantileTransformer)

## 🧪 Testing

Run the test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=stripje
```

## 📁 Examples

See the `examples/` directory for more comprehensive examples and benchmarks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
