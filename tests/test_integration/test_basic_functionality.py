import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stripje


def main():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))]
    )

    pipeline.fit(X, y)
    fast_predict = stripje.compile_pipeline(pipeline)

    test_row = X[0].tolist()
    original_pred = pipeline.predict([test_row])[0]
    fast_pred = fast_predict(test_row)

    # Ensure same value (cast to int to avoid numpy scalar mismatch)
    assert int(original_pred) == int(fast_pred)

    print(
        f"Basic functionality test passed on {os.getenv('RUNNER_OS', 'unknown')} with Python {sys.version.split()[0]}"
    )


if __name__ == "__main__":
    main()
