#!/usr/bin/env python3
"""Local demo only. Same prediction logic as Kaggle inference notebook."""
import argparse, joblib
from name_that_llm.io import load_test
from name_that_llm.submission import write_submission
p=argparse.ArgumentParser()
p.add_argument('--model',default='artifacts/baseline/model.joblib')
p.add_argument('--test',default='data/raw/test.csv')
p.add_argument('--output',default='reports/local_demo_submission.parquet')
a=p.parse_args()
test=load_test(a.test)
model=joblib.load(a.model)   # trust only artifacts you trained yourself
submission=write_submission(test,model.predict(test.text.astype(str).tolist()),a.output)
print(f'Wrote {len(submission)} rows -> {a.output}')
