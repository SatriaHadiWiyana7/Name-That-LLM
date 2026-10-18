import copy
import io
import json
from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from name_that_llm.baseline import ordered_probabilities
from name_that_llm.labels import LABELS
from name_that_llm.stylometry import (
    FEATURE_NAMES,
    StylometryExtractor,
    feature_schema,
    make_stylometry,
    validate_stylometry_config,
)


def config():
    path = Path(__file__).resolve().parents[1] / "configs/p3_stylometry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def features(text):
    return dict(zip(FEATURE_NAMES, StylometryExtractor().transform([text])[0]))


def test_schema_has_fixed_unique_order_and_definitions():
    schema = feature_schema()
    assert schema["schema_version"] == 1
    assert schema["feature_count"] == len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 81
    assert schema["unicode_database_version"] == unicodedata.unidata_version
    assert tuple(item["name"] for item in schema["features"]) == FEATURE_NAMES
    assert all(item["definition"] for item in schema["features"])
    assert FEATURE_NAMES[:3] == ("text_length", "utf8_byte_length", "word_count")
    assert FEATURE_NAMES[41::2] == (
        "period_count", "comma_count", "semicolon_count", "colon_count",
        "exclamation_count", "question_count", "hyphen_count", "em_dash_count",
        "en_dash_count", "apostrophe_count", "quote_count", "curly_single_quote_count",
        "curly_double_quote_count", "ellipsis_count", "parenthesis_count", "bracket_count",
        "slash_count", "backslash_count", "underscore_count", "bullet_count",
    )
    np.testing.assert_array_equal(StylometryExtractor().get_feature_names_out(), FEATURE_NAMES)
    schema["features"].clear()
    assert len(feature_schema()["features"]) == 81


def test_empty_text_and_empty_batch_are_finite_float64():
    extractor = StylometryExtractor()
    output = extractor.fit_transform([""])
    assert output.dtype == np.float64
    assert output.shape == (1, 81)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output, np.zeros((1, 81)))
    assert extractor.transform([]).shape == (0, 81)


def test_word_sentence_and_lexical_statistics_use_preregistered_definitions():
    text = "The cat can't jump. THE cats can\u2019t."
    result = features(text)
    assert result["word_count"] == 7
    lengths = [3, 3, 5, 4, 3, 4, 5]
    assert result["word_length_mean"] == pytest.approx(27 / 7)
    assert result["word_length_std"] == pytest.approx(np.std(lengths, ddof=0))
    assert result["word_length_min"] == 3
    assert result["word_length_max"] == 5
    assert result["sentence_count"] == 2
    assert result["sentence_word_count_mean"] == 3.5
    assert result["sentence_word_count_std"] == 0.5
    assert result["sentence_word_count_min"] == 3
    assert result["sentence_word_count_max"] == 4
    assert result["unique_word_count"] == 6
    assert result["type_token_ratio"] == pytest.approx(6 / 7)
    assert result["hapax_token_rate"] == pytest.approx(5 / 7)
    assert result["short_token_rate"] == pytest.approx(3 / 7)
    assert result["long_token_rate"] == 0
    assert result["apostrophe_token_rate"] == pytest.approx(2 / 7)
    assert result["function_word_rate"] == pytest.approx(2 / 7)


def test_casefolding_changes_types_but_never_original_lengths():
    result = features("Stra\u00dfe STRASSE")
    assert result["word_count"] == 2
    assert result["unique_word_count"] == 1
    assert result["hapax_token_rate"] == 0
    assert result["type_token_ratio"] == 0.5
    assert result["word_length_mean"] == 6.5
    assert result["long_token_rate"] == 0.5


def test_sentence_heuristic_ignores_wordless_segments_and_counts_runs_once():
    result = features("... !?? 123. Hi there!!! A? _\n")
    assert result["sentence_count"] == 2
    assert result["sentence_word_count_mean"] == 1.5
    assert result["sentence_word_count_std"] == 0.5
    assert result["word_count"] == 3


def test_unicode_categories_digits_and_whitespace_use_original_codepoints():
    text = "A\u00e9\u03a9\u4e2d\u00df\uff11\u00b2\U0001f642e\u0301\u2014\t\n"
    result = features(text)
    assert result["text_length"] == 13
    assert result["utf8_byte_length"] == 27
    expected = {
        "whitespace": 2, "newline": 1, "tab": 1, "alphabetic": 6,
        "uppercase": 2, "lowercase": 3, "digit": 2, "non_ascii": 9,
        "unicode_punctuation": 1, "unicode_symbol": 1, "unicode_mark": 1,
    }
    for name, count in expected.items():
        assert result[f"{name}_count"] == count
        assert result[f"{name}_rate"] == pytest.approx(count / 13)
    # No NFC normalization: decomposed e + combining mark occupies two points.
    assert features("e\u0301")["text_length"] == 2
    assert features("\u00e9")["text_length"] == 1


def test_punctuation_groups_count_individual_characters_and_use_total_length():
    text = ".,;:!?-\u2014\u2013'\"\u2018\u2019\u201c\u201d\u2026()[]/\\_\u2022 \n"
    result = features(text)
    paired = {"curly_single_quote", "curly_double_quote", "parenthesis", "bracket"}
    for name in FEATURE_NAMES[41::2]:
        prefix = name.removesuffix("_count")
        count = 2 if prefix in paired else 1
        assert result[name] == count
        assert result[f"{prefix}_rate"] == pytest.approx(count / len(text))
    result = features("...\u2026")
    assert result["period_count"] == 3
    assert result["ellipsis_count"] == 1
    assert result["sentence_count"] == 0


@pytest.mark.parametrize("value", [None, 3, np.nan, b"bytes", ["nested"]])
def test_nonstring_items_are_rejected_without_coercion(value):
    extractor = StylometryExtractor()
    with pytest.raises(TypeError, match="string"):
        extractor.fit(["valid", value])
    with pytest.raises(TypeError, match="string"):
        extractor.transform([value])


@pytest.mark.parametrize("batch", ["single text", b"bytes", None, 4])
def test_noniterables_and_single_string_are_not_treated_as_documents(batch):
    with pytest.raises(TypeError, match="string"):
        StylometryExtractor().transform(batch)


def test_extractor_has_no_corpus_statistics_and_is_batch_order_invariant():
    extractor = StylometryExtractor()
    text = "THE same text. 42\n"
    before = extractor.transform([text])
    extractor.fit(["training " * 1000, "other\u2014style"], [0, 1])
    assert vars(extractor) == {}
    np.testing.assert_array_equal(before, extractor.transform(["unrelated", text])[1:])
    np.testing.assert_array_equal(before, extractor.fit_transform([text]))
    np.testing.assert_array_equal(before, clone(extractor).transform([text]))


def test_fit_transform_handles_single_use_iterators_without_consuming_twice():
    texts = ["First text!", "Second\u2014text.", ""]
    extractor = StylometryExtractor()
    expected = extractor.transform(texts)
    np.testing.assert_array_equal(extractor.fit_transform(iter(texts)), expected)
    np.testing.assert_array_equal(extractor.transform(iter(texts)), expected)
    assert vars(extractor) == {}


def test_pipeline_scaler_fits_only_training_rows_and_serializes_exactly():
    train = ["a " * (i + 1) + "!" * (i % 3) + str(i) for _ in range(2) for i in range(18)]
    targets = LABELS * 2
    validation = ["Z" * 10000 + " 9" * 1000]
    cfg = config()
    original = copy.deepcopy(cfg)
    pipeline = make_stylometry(cfg)
    assert list(pipeline.named_steps) == ["features", "scale", "clf"]
    pipeline.fit(train, targets)
    raw_train = StylometryExtractor().transform(train)
    scaler = pipeline["scale"]
    assert scaler.n_samples_seen_ == len(train)
    np.testing.assert_allclose(scaler.mean_, raw_train.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, raw_train.var(axis=0))
    assert np.max(np.abs(scaler.mean_ - StylometryExtractor().transform(train + validation).mean(axis=0))) > 100
    saved_mean = scaler.mean_.copy()
    probabilities = ordered_probabilities(pipeline, validation)
    assert probabilities.shape == (1, 18)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(scaler.mean_, saved_mean)
    assert scaler.n_samples_seen_ == len(train)
    assert pipeline["clf"].get_params()["random_state"] == 42
    assert pipeline["clf"].get_params()["C"] == 3.0
    assert cfg == original
    fresh = clone(pipeline)
    assert not hasattr(fresh["scale"], "mean_")
    payload = io.BytesIO()
    joblib.dump(pipeline, payload)
    payload.seek(0)
    restored = joblib.load(payload)
    np.testing.assert_array_equal(ordered_probabilities(restored, validation), probabilities)
    np.testing.assert_array_equal(restored["features"].get_feature_names_out(), FEATURE_NAMES)


@pytest.mark.parametrize("field,value", [
    ("random_seed", 43), ("n_splits", 4), ("features", {"schema_version": 2}),
    ("scaler", {"with_mean": False, "with_std": True}),
    ("classifier", {"C": 4}), ("extra_setting", True),
    ("features", {"schema_version": True}),
    ("scaler", {"with_mean": 1, "with_std": 1}),
])
def test_unregistered_config_changes_fail_before_pipeline_creation(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match="P3"):
        validate_stylometry_config(cfg)
    with pytest.raises(ValueError, match="P3"):
        make_stylometry(cfg)


def test_config_is_exact_preregistered_model():
    cfg = config()
    validate_stylometry_config(cfg)
    model = make_stylometry(cfg)
    assert model["scale"].with_mean is True
    assert model["scale"].with_std is True
    expected = {"C": 3.0, "max_iter": 1200, "class_weight": "balanced", "solver": "lbfgs", "tol": 1e-4, "random_state": 42}
    assert {name: model["clf"].get_params()[name] for name in expected} == expected
