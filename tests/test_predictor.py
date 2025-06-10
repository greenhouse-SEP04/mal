from ml_service.predictor import SoilMoisturePredictor

def test_forecast_shape(monkeypatch):
    from ml_service import config
    monkeypatch.setattr(config,"TOTAL_STEPS",10)
    monkeypatch.setattr(config,"MODEL_6H_STEPS",5)
    pred=SoilMoisturePredictor()
    pred._MODEL=lambda x:x
    class Scaler:
        @staticmethod
        def transform(x): return x
        @staticmethod
        def inverse_transform(x): return x
    pred._SX=pred._SY=Scaler()
    sample={"Soil":40,"Temperature":24,"Humidity":55}
    fc=pred.forecast(sample)
    assert len(fc)==10
    assert all(v>=0 for v in fc)