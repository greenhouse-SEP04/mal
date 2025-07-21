import pandas as pd
import pytest
from src.greenhouse_ml_service import Preprocessor, _detect_feature_types


def test_detect_feature_types():
    df = pd.DataFrame({
        'a':[1,2,3], 'b':['x','y','z'], 'target':[0,1,0]
    })
    num, cat = _detect_feature_types(df, target='target')
    assert 'a' in num and 'b' in cat


def test_preprocessor_build():
    df = pd.DataFrame({'num':[1, None, 3], 'cat':['a','b',None], 'target':[0,1,0]})
    pre = Preprocessor(numeric_strategy='median').build(df, 'target')
    X = pre.fit_transform(df.drop(columns=['target']))
    # should have no missing values
    assert not pd.isnull(X).any()