#!/usr/bin/env python3
"""Independently verify two saved baseline runs and smoke-test the Kaggle template."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from name_that_llm.analysis import validate_oof
from name_that_llm.io import load_test, sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.metrics import classification_report
from name_that_llm.runtime import ResourceMonitor
from name_that_llm.submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-manifest', required=True)
    parser.add_argument('--second-manifest', required=True)
    parser.add_argument('--test', default='data/raw/test.csv')
    parser.add_argument('--local-submission', required=True)
    parser.add_argument('--template', default='kaggle/inference_template.py')
    parser.add_argument('--smoke-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.smoke_output).exists():
        parser.error('Verification outputs already exist; choose new paths')
    with ResourceMonitor() as monitor:
        manifests, metrics, frames = [], [], []
        for path in (args.first_manifest, args.second_manifest):
            manifest = json.loads(Path(path).read_text(encoding='utf-8'))
            for artifact in manifest['artifacts'].values():
                assert sha256_file(artifact['path']) == artifact['sha256'], 'Artifact hash mismatch'
            assert sha256_file(manifest['dataset_path']) == manifest['dataset_sha256']
            assert manifest['labels'] == LABELS
            metric = json.loads(Path(manifest['artifacts']['metrics']['path']).read_text())
            oof = pd.read_csv(manifest['artifacts']['oof']['path'], dtype={'id': str, 'prompt_id': str})
            validate_oof(oof)
            recomputed = classification_report(oof.true_model, oof.pred_model)
            for key in ('balanced_accuracy', 'class_recall', 'class_support', 'confusion_matrix'):
                assert metric[key] == recomputed[key], f'Stored OOF {key} mismatch'
            for fold in metric['folds']:
                subset = oof[oof.fold == fold['fold']]
                measured = classification_report(subset.true_model, subset.pred_model)
                for key in ('balanced_accuracy', 'class_recall', 'confusion_matrix'):
                    assert fold[key] == measured[key], f'Stored fold {key} mismatch'
            manifests.append(manifest)
            metrics.append(metric)
            frames.append(oof)
        first, second = manifests
        for key in ('dataset_sha256', 'fold_assignments_sha256', 'random_seed', 'config', 'labels'):
            assert first[key] == second[key], f'Reproduction {key} differs'
        metadata_columns = ['id', 'prompt_id', 'true_model', 'pred_model', 'fold']
        pd.testing.assert_frame_equal(frames[0][metadata_columns], frames[1][metadata_columns])
        pcols = ['p_' + label for label in LABELS]
        p1, p2 = frames[0][pcols].to_numpy(), frames[1][pcols].to_numpy()
        maximum_difference = float(np.abs(p1 - p2).max())
        assert np.allclose(p1, p2, rtol=0, atol=1e-12), 'Repeated probabilities differ'
        assert metrics[0]['confusion_matrix'] == metrics[1]['confusion_matrix']
        assert metrics[0]['balanced_accuracy'] == metrics[1]['balanced_accuracy']
        assert first['model_sha256'] == second['model_sha256'], 'Serialized models differ'

        # Execute the actual template, changing only its three deployment paths.
        # No fixed predictions are supplied; the template loads and predicts test.csv.
        replacements = {'COMP_PATH': str(Path(args.test).resolve()),
                        'ARTIFACT_PATH': first['artifacts']['model']['path'],
                        'OUT_PATH': str(Path(args.smoke_output).resolve())}
        tree = ast.parse(Path(args.template).read_text(encoding='utf-8'))
        changed = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in replacements:
                    node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                          args=[ast.Constant(replacements[name])], keywords=[])
                    changed.add(name)
        assert changed == set(replacements), 'Template deployment constants not found'
        exec(compile(ast.fix_missing_locations(tree), args.template, 'exec'), {'__name__': '__main__'})
        test = load_test(args.test)
        smoke = pd.read_parquet(args.smoke_output)
        local = pd.read_parquet(args.local_submission)
        validate_submission(test, smoke)
        validate_submission(test, local)
        pd.testing.assert_frame_equal(smoke.reset_index(drop=True), local.reset_index(drop=True))
    result = {
        'utc_created': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'argv': [sys.executable, *sys.argv], 'source_sha256': sha256_file(__file__),
        'manifest_sha256': {path: sha256_file(path) for path in (args.first_manifest, args.second_manifest)},
        'rows': len(frames[0]), 'prompts': int(frames[0].prompt_id.nunique()),
        'balanced_accuracy': metrics[0]['balanced_accuracy'], 'delta_balanced_accuracy': 0.0,
        'all_artifact_hashes_verified': True, 'all_stored_metrics_recomputed': True,
        'same_dataset_folds_seed_config': True, 'same_predictions_and_confusion': True,
        'max_probability_abs_difference': maximum_difference, 'probability_tolerance': 1e-12,
        'serialized_model_hashes_equal': True, 'model_sha256': first['model_sha256'],
        'template_smoke': {'rows': len(smoke), 'schema_valid': True, 'matches_local_predictions': True,
                           'template_sha256': sha256_file(args.template),
                           'test_sha256': sha256_file(args.test), 'output_sha256': sha256_file(args.smoke_output),
                           'scope': 'Local path-substituted template on sample test, not Kaggle hidden rerun'},
        'resources': monitor.report(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
