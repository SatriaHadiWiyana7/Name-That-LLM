#!/usr/bin/env python3
import argparse
import pandas as pd
from name_that_llm.io import load_test
from name_that_llm.submission import validate_submission
p=argparse.ArgumentParser()
p.add_argument('--test',required=True)
p.add_argument('--submission',required=True)
a=p.parse_args()
test=load_test(a.test)
sub=pd.read_parquet(a.submission)
validate_submission(test,sub)
print(f'VALID parquet submission, rows={len(sub)}, labels_used={sub.model.nunique()}')
