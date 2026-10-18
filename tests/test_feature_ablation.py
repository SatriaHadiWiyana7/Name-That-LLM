import copy
import json
from pathlib import Path

import numpy as np
import pytest

from name_that_llm.baseline import make_baseline, ordered_probabilities
from name_that_llm.labels import LABELS


def baseline_config():
    path = Path(__file__).resolve().parents[1] / 'configs/baseline.json'
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize('group', ['word', 'char'])
def test_ablation_only_deletes_feature_block_and_keeps_parameters(group):
    original = baseline_config()
    config = copy.deepcopy(original)
    config['feature_groups'] = [group]
    before = copy.deepcopy(config)
    combined = make_baseline(original)
    ablated = make_baseline(config)
    assert config == before
    retained = dict(ablated['features'].transformer_list)
    assert list(retained) == [group]
    assert retained[group].get_params() == dict(combined['features'].transformer_list)[group].get_params()
    assert ablated['clf'].get_params() == combined['clf'].get_params()


@pytest.mark.parametrize('group', ['word', 'char'])
def test_single_block_preserves_representation_and_validation_isolation(group):
    config = baseline_config()
    config['feature_groups'] = [group]
    config[group]['min_df'] = 1
    texts = [f'author{i} style sample; distinctive words — {i}.' for _ in range(2) for i in range(18)]
    model = make_baseline(config)
    model.fit(texts, LABELS * 2)
    held_out = ['validationexclusiveword ŽŽŽ; unseen prompt!']
    vectorizer = dict(model['features'].transformer_list)[group]
    assert 'validationexclusiveword' not in vectorizer.vocabulary_
    assert 'ŽŽŽ' not in vectorizer.vocabulary_
    union_features = model['features'].transform(held_out)
    direct_features = vectorizer.transform(held_out)
    assert (union_features != direct_features).nnz == 0
    probabilities = ordered_probabilities(model, held_out)
    assert probabilities.shape == (1, 18)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


@pytest.mark.parametrize('groups', [[], ['invalid'], ['char', 'word'], ['word', 'word'], 'word'])
def test_invalid_or_reordered_feature_blocks_rejected(groups):
    config = baseline_config()
    config['feature_groups'] = groups
    with pytest.raises(ValueError, match='feature_groups'):
        make_baseline(config)


def test_configs_change_only_name_and_feature_selection():
    original = baseline_config()
    root = Path(__file__).resolve().parents[1]
    for group in ('word', 'char'):
        config = json.loads((root / f'configs/p2_{group}_only.json').read_text())
        assert config.pop('feature_groups') == [group]
        config['name'] = original['name']
        assert config == original
