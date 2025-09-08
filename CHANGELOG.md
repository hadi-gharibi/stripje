# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Placeholder for future features

### Changed
- Placeholder for future changes

### Deprecated
- Placeholder for future deprecations

### Removed
- Placeholder for future removals

### Fixed
- Placeholder for future fixes

### Security
- Placeholder for future security updates

## [0.1.0] - 2025-09-07

### Added
- Initial release of Stripje - Fast Pipeline Compiler
- Core pipeline compilation functionality for scikit-learn pipelines
- Support for major preprocessing transformers:
  - StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
  - Normalizer, OneHotEncoder, OrdinalEncoder, LabelEncoder
  - QuantileTransformer, SelectKBest
- Support for estimators:
  - LogisticRegression, LinearRegression
  - RandomForestClassifier, DecisionTreeClassifier
  - GaussianNB
- Support for ColumnTransformer with recursive compilation
- Registry system for extending transformer support
- Comprehensive test suite
- Performance benchmarking examples
- MIT License
- Python 3.9+ compatibility

### Performance
- 2-10x speedup for single-row inference compared to standard scikit-learn pipelines
- Optimized single-row functions that avoid numpy overhead

[Unreleased]: https://github.com/hadi-gharibi/sklean/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hadi-gharibi/sklean/releases/tag/v0.1.0
