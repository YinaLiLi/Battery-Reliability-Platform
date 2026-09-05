import joblib
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def test_pinned_model_roundtrip(tmp_path):
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = Surv.from_arrays([True, False] * 10, np.arange(1, 21, dtype=float))
    model = RandomSurvivalForest(n_estimators=2, min_samples_leaf=2, random_state=0).fit(x, y)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict(x), joblib.load(path).predict(x))
