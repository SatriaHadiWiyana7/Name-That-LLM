#!/usr/bin/env python3
"""Append external score updates. Does not fabricate score ownership, leaderboard or rank."""
import argparse, csv
from pathlib import Path
from datetime import datetime,timezone
fields=['timestamp_utc','source','split','model_id','score','rank','cutoff_kind','notes']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--score',type=float)
p.add_argument('--source',default='user_reported',choices=['user_reported','own_submission','leaderboard_cutoff','local_oof'])
p.add_argument('--split',default='unknown',choices=['unknown','public','private','oof'])
p.add_argument('--model-id',default='')
p.add_argument('--rank',type=int)
p.add_argument('--cutoff-kind',default='',choices=['','top5','top10'])
p.add_argument('--notes',default='')
p.add_argument('--file',default='reports/score_updates.csv')
p.add_argument('--report',action='store_true')
a=p.parse_args()
path=Path(a.file);path.parent.mkdir(parents=True,exist_ok=True)
if a.score is not None:
 if not 0<=a.score<=1: raise SystemExit('Balanced accuracy should be in [0,1].')
 if a.source=='leaderboard_cutoff' and not a.cutoff_kind:
  raise SystemExit('cutoff reports require --cutoff-kind top5/top10')
 if a.source!='leaderboard_cutoff' and a.cutoff_kind:
  raise SystemExit('Only leaderboard_cutoff can have --cutoff-kind')
 exists=path.exists()
 with path.open('a',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields)
  if not exists:writer.writeheader()
  writer.writerow({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':a.source,'split':a.split,'model_id':a.model_id,'score':f'{a.score:.6f}','rank':a.rank if a.rank is not None else '', 'cutoff_kind':a.cutoff_kind,'notes':a.notes})
 print('Recorded:',a.score, '| source:',a.source,'| split:',a.split)
if a.report:
 if not path.exists(): raise SystemExit('No score log yet.')
 with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 print(f'Score events: {len(rows)}')
 for row in rows[-10:]:
  print(row['timestamp_utc'][:19],row['source'],row['split'],'score=',row['score'],'rank=',row['rank'] or '?','model=',row['model_id'] or '?')
 # Never compare OOF vs leaderboard or user_reported unknown.
 for split in ('public','private'):
  own=[r for r in rows if r['split']==split and r['source']=='own_submission']
  if not own:continue
  recent=own[-1]
  print(f'Latest OWN {split} score: {recent["score"]}  rank: {recent["rank"] or "unknown"}')
  for kind in ('top5','top10'):
   thresholds=[r for r in rows if r['split']==split and r['source']=='leaderboard_cutoff' and r['cutoff_kind']==kind]
   if thresholds:
    diff=float(thresholds[-1]['score'])-float(recent['score'])
    print(f'  Approx {kind} cutoff gap ({split}, timestamps may differ): {diff:+.4f} BA points; + means below cutoff')
