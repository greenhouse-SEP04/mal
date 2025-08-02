import pandas as pd
from src.handler import _fit_model, handler


def test_fit_and_predict_regression(tmp_path, monkeypatch):
    # synthetic regression
    df = pd.DataFrame({'x':[1,2,3,4,5],'y':[2,4,6,8,10]})
    model, metrics = _fit_model(df, 'y', 'ridge')
    assert 'rmse' in metrics and metrics['r2'] > 0.99


def test_handler_predict(tmp_path, monkeypatch):
    # train a small model then invoke predict
    df = pd.DataFrame({'x':[1,2,3,4],'y':[1,2,3,4]})
    model, _ = _fit_model(df, 'y', 'logreg')
    buf = tmp_path / 'model.joblib'
    import joblib
    joblib.dump(model, buf)
    # upload locally
    uri = f"file://{buf}"
    resp = handler({'action':'predict','s3_uri':uri,'payload':[{'x':5}]}, None)
    assert resp['status']=='predicted'
    assert isinstance(resp['predictions'][0], (int,))