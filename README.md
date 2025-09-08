# Stripje - Make sklearn pipelines lean

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/hadi-gharibi/stripje)

**Speed up your scikit-learn pipelines for single-row predictions by 2-10x!**

Stripje is a high-performance compiler that converts trained scikit-learn pipelines into optimized Python functions, eliminating numpy overhead for single-row inference.

## 🚀 Why Stripje?

- **⚡ 2-200x faster** single-row predictions, depending on the pipeline complexity
- **🔧 Drop-in replacement** - works with your existing pipelines
- **🎯 Zero configuration** - just compile and use
- **🛠️ Production ready** - optimized for real-time inference

## 📦 Installation

```bash
pip install stripje
```

Or with uv (recommended):
```bash
uv add stripje
```

## ⚡ Quick Start

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stripje import compile_pipeline

# 1. Create and fit your pipeline as usual
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# 2. Compile for fast single-row inference
fast_predict = compile_pipeline(pipeline)

# 3. Get predictions up to 10x faster!
test_row = [1.2, -0.5, 0.8, -1.1]
prediction = fast_predict(test_row)  # Much faster than pipeline.predict([test_row])
```

## 🎯 The Problem We Solve

**Standard scikit-learn pipelines are slow for single predictions** because they're optimized for batch processing. When you need to predict one row at a time (like in web APIs), numpy operations create unnecessary overhead.

**Stripje compiles your trained pipeline** into a specialized function that:
- ✅ Extracts fitted parameters once
- ✅ Eliminates array creation overhead  
- ✅ Uses native Python operations
- ✅ Maintains identical results

## 📊 Performance Comparison

```python
import time
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Setup
X, y = make_classification(n_samples=1000, n_features=20)
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipeline.fit(X, y)
fast_predict = compile_pipeline(pipeline)

test_row = X[0].tolist()

# Benchmark single-row predictions
def benchmark_standard():
    start = time.time()
    for _ in range(1000):
        pipeline.predict([test_row])
    return time.time() - start

def benchmark_compiled():
    start = time.time()
    for _ in range(1000):
        fast_predict(test_row)
    return time.time() - start

standard_time = benchmark_standard()
compiled_time = benchmark_compiled()
speedup = standard_time / compiled_time

print(f"Standard pipeline: {standard_time:.3f}s")
print(f"Compiled pipeline: {compiled_time:.3f}s") 
print(f"Speedup: {speedup:.1f}x faster!")
```

## 🔧 Supported Components

Stripje supports the most commonly used scikit-learn components:

### 🔄 Transformers
- **Scalers**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`
- **Encoders**: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`  
- **Other**: `Normalizer`, `QuantileTransformer`, `SelectKBest`

### 🎯 Estimators  
- **Classification**: `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `GaussianNB`
- **Regression**: `LinearRegression`

### 🏗️ Composite
- **`ColumnTransformer`** - Full support with nested compilation

*More components coming soon! See [Contributing](#-contributing) to request or add support.*

## 📖 More Examples

### Complex Pipeline with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Create a complex pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['category', 'region'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10))
])

# Fit and compile
pipeline.fit(X_train, y_train)
fast_predict = compile_pipeline(pipeline)

# Single-row prediction
row = [25, 50000, 'A', 'North']  # [age, income, category, region]
prediction = fast_predict(row)
```

### Real-World API Usage

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load and compile your model once at startup
model = joblib.load('trained_pipeline.pkl')
fast_predict = compile_pipeline(model)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = fast_predict(data)  # Super fast!
    return jsonify({'prediction': prediction.tolist()})
```

## 🚫 Limitations

- Input must be lists/arrays (no pandas DataFrames directly)
- No sparse matrix support
- Some transformers use approximations (e.g., `QuantileTransformer`)
- Only listed components are supported

## 📚 API Reference

### `compile_pipeline(pipeline)`
Compiles a fitted scikit-learn pipeline into a fast single-row prediction function.

**Args:**
- `pipeline`: A fitted scikit-learn Pipeline

**Returns:**  
- Function that takes a single row (list/array) and returns predictions

**Raises:**
- `ValueError`: If pipeline contains unsupported components

### `get_supported_transformers()`
Returns list of all supported transformer/estimator classes.

## 📁 Examples & Benchmarks

Check out the `examples/` directory for:
- **`simple_example.py`** - Basic usage
- **`benchmark.py`** - Performance comparisons  
- **`comprehensive_benchmark.py`** - Detailed benchmarks
- **`profiler_demo.py`** - Profiling tools

## 🔌 Extending Support

Want to add support for a new transformer? It's easy:

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🛠️ Development

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

- **Ruff**: Fast linting, formatting, and import sorting
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

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=stripje
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
