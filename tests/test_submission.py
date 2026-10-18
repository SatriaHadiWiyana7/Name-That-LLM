import pandas as pd
import pytest
from name_that_llm.submission import validate_submission, write_submission


def test_validate_valid():
    test=pd.DataFrame({'id':['a','b'],'text':['one','two']})
    submission=pd.DataFrame({'id':['a','b'],'model':['gpt-5.5','kimi-k3']})
    validate_submission(test,submission)


def test_reject_duplicate_and_illegal():
    test=pd.DataFrame({'id':['a','b'],'text':['one','two']})
    with pytest.raises(ValueError):
        validate_submission(test,pd.DataFrame({'id':['a','a'],'model':['gpt-5.5','kimi-k3']}))
    with pytest.raises(ValueError):
        validate_submission(test,pd.DataFrame({'id':['a','b'],'model':['unknown','kimi-k3']}))


def test_parquet_round_trip_preserves_ids_and_labels(tmp_path):
    test = pd.DataFrame({'id': ['0001', '0002'], 'text': ['a', 'b']})
    expected = write_submission(test, ['gpt-5.5', 'kimi-k3'], tmp_path / 'submission.parquet')
    actual = pd.read_parquet(tmp_path / 'submission.parquet')
    pd.testing.assert_frame_equal(actual, expected)
    validate_submission(test, actual)
