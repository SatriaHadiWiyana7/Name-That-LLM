#!/usr/bin/env python3
import argparse, json, re, sys, hashlib
from datetime import datetime, timezone
from zipfile import ZipFile
from pathlib import Path
from name_that_llm.io import load_train, load_test, sha256_file
from name_that_llm.labels import LABELS

p = argparse.ArgumentParser()
p.add_argument('--train',default='data/raw/train.csv')
p.add_argument('--test',default='data/raw/test.csv')
p.add_argument('--archive', help='Original user-owned ZIP for byte comparison')
p.add_argument('--output', help='Write aggregate JSON, without passages')
a=p.parse_args()
tr=load_train(a.train)
test=load_test(a.test)
counts=tr.groupby('prompt_id').model.nunique()
observed=set(map(tuple,tr[['prompt_id','model']].itertuples(index=False,name=None)))
missing=[(pid,lbl) for pid in sorted(tr.prompt_id.unique()) for lbl in LABELS if (pid,lbl) not in observed]
words=tr.text.map(lambda s:len(re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", s, re.UNICODE)))
result={'train_rows':len(tr),'test_example_rows':len(test),'num_prompts':int(tr.prompt_id.nunique()),'labels':len(tr.model.unique()),'missing_prompt_model_pairs':missing,'complete_prompts':int((counts==18).sum()),'word_mean':round(float(words.mean()),2),'word_median':float(words.median()),'word_min':int(words.min()),'word_max':int(words.max()),'duplicate_text':int(tr.text.duplicated().sum())}
spread = words.groupby(tr.prompt_id).agg(['min', 'max'])
punctuation = {'em_dash': '—', 'semicolon': ';', 'colon': ':',
               'newline': '\n', 'tab': '\t', 'nonascii': r'[^\x00-\x7f]'}
result.update({
    'utc_created': datetime.now(timezone.utc).isoformat(),
    'argv': [sys.executable, *sys.argv], 'source_sha256': sha256_file(__file__),
    'dataset_sha256': sha256_file(a.train), 'test_sha256': sha256_file(a.test),
    'label_order': LABELS, 'class_counts': {l: int((tr.model == l).sum()) for l in LABELS},
    'prompt_response_counts': {str(k): int(v) for k, v in counts.value_counts().items()},
    'below_40_words': int((words < 40).sum()), 'above_250_words': int((words > 250).sum()),
    'mean_prompt_word_spread': round(float((spread['max'] - spread['min']).mean()), 2),
    'word_regex': r"\b[\w]+(?:['’\-][\w]+)*\b",
    'duplicate_train_ids': int(tr.id.duplicated().sum()),
    'duplicate_prompt_model': int(tr[['prompt_id', 'model']].duplicated().sum()),
    'null_train_fields': {k: int(v) for k, v in tr.isna().sum().items()},
    'train_test_id_overlap': len(set(tr.id) & set(test.id)),
    'nontrimmed_train_text': int((tr.text != tr.text.str.strip()).sum()),
    'punctuation_presence_pct': {k: round(float(tr.text.str.contains(v).mean() * 100), 2)
                                 for k, v in punctuation.items()},
})
if a.archive:
    with ZipFile(a.archive) as archive:
        members = {Path(n).name: n for n in archive.namelist() if not n.endswith('/')}
        hashes = {name: hashlib.sha256(archive.read(members[name])).hexdigest()
                  for name in ('train.csv', 'test.csv', 'sample_submission.parquet')}
    result['original_archive'] = {
        'sha256': sha256_file(a.archive), 'members_sha256': hashes,
        'train_matches': hashes['train.csv'] == result['dataset_sha256'],
        'test_matches': hashes['test.csv'] == result['test_sha256'],
    }
if a.output:
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
print(json.dumps(result,indent=2))
