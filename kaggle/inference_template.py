"""COPY TO KAGGLE NOTEBOOK CELL. Training weights uploaded as private input.
This template uses scikit-learn pipeline serialized by our own train_baseline.py.
Edit ARTIFACT_PATH after attaching your model input dataset.
"""
from pathlib import Path
import joblib
import pandas as pd

COMP_PATH=Path('/kaggle/input/competitions/name-that-llm/test.csv')
ARTIFACT_PATH=Path('/kaggle/input/REPLACE_WITH_YOUR_PRIVATE_MODEL_DATASET/model.joblib')
OUT_PATH=Path('/kaggle/working/submission.parquet')

LABELS={
    'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
    'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
    'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra',
    'gpt-6-astra','kimi-k3','qwen3.8-max',
}
assert COMP_PATH.exists(), 'Competition input missing; ensure notebook is attached to competition.'
assert ARTIFACT_PATH.exists(), f'Attach private model dataset and edit ARTIFACT_PATH: {ARTIFACT_PATH}'

test=pd.read_csv(COMP_PATH,dtype={'id':str,'text':str})
assert test.id.is_unique and test.id.notna().all() and test.text.notna().all()
model=joblib.load(ARTIFACT_PATH)  # only trusted own model
pred=model.predict(test.text.astype(str).tolist())
submission=pd.DataFrame({'id':test.id,'model':pred})
assert submission.id.is_unique and submission.model.notna().all()
assert set(submission.id)==set(test.id) and len(submission)==len(test)
assert set(submission.model).issubset(LABELS)
OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
submission.to_parquet(OUT_PATH,index=False)
print('submission.parquet rows:',len(submission), 'unique labels:',submission.model.nunique())
