import pandas as pd
from name_that_llm.validation import group_folds, fold_assignments


def test_prompt_groups_are_disjoint():
    labels=['A','B','C']
    df=pd.DataFrame([{'id':f'{p}-{m}','prompt_id':f'p{p:03}','text':'x','model':m} for p in range(15) for m in labels])
    got=[]
    for fold,tr,va in group_folds(df,5):
        assert not set(df.iloc[tr].prompt_id) & set(df.iloc[va].prompt_id)
        got.extend(va.tolist())
    assert sorted(got)==list(range(len(df)))
    assert set(fold_assignments(df,5))==set(range(5))
