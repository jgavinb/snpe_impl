import numpy as np
from snpe_impl.estimator import BaggedNN


def test_bagged_nn():
    X = np.random.randn(100, 2)
    y = X[:, 0] + np.random.randn(100) * 0.1
    model = BaggedNN(m=5, num_bags=10).fit(X, y)
    pred = model.predict(X[:5])
    assert np.allclose(pred, y[:5], atol=0.5), (
        "Predictions should approximate true function"
    )
