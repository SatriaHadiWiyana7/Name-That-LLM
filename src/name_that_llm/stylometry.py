"""The fixed, stateless 81-feature P3 stylometry specification.

All statistics use one original string at a time. No corpus-level information
is learned here; the returned pipeline learns scaling only during ``fit``.
"""

from collections import Counter
from copy import deepcopy
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = 1
WORD_PATTERN = r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*"
_WORDS = re.compile(WORD_PATTERN)
_SENTENCES = re.compile(r"[.!?]+")
FUNCTION_WORDS = frozenset(
    "a an the and or but if then because as of to in on at by for from with "
    "without is are was were be been being it its this that these those i we "
    "you he she they them our your their".split()
)

_CHARACTER_CATEGORIES = (
    ("whitespace", "Python str.isspace()"),
    ("newline", "U+000A"),
    ("tab", "U+0009"),
    ("alphabetic", "Python str.isalpha()"),
    ("uppercase", "Python str.isupper()"),
    ("lowercase", "Python str.islower()"),
    ("digit", "Python str.isdigit()"),
    ("non_ascii", "code point greater than 127"),
    ("unicode_punctuation", "Unicode general category starting with P"),
    ("unicode_symbol", "Unicode general category starting with S"),
    ("unicode_mark", "Unicode general category starting with M"),
)
_PUNCTUATION = (
    ("period", "."),
    ("comma", ","),
    ("semicolon", ";"),
    ("colon", ":"),
    ("exclamation", "!"),
    ("question", "?"),
    ("hyphen", "-"),
    ("em_dash", "\u2014"),
    ("en_dash", "\u2013"),
    ("apostrophe", "'"),
    ("quote", '"'),
    ("curly_single_quote", "\u2018\u2019"),
    ("curly_double_quote", "\u201c\u201d"),
    ("ellipsis", "\u2026"),
    ("parenthesis", "()"),
    ("bracket", "[]"),
    ("slash", "/"),
    ("backslash", "\\"),
    ("underscore", "_"),
    ("bullet", "\u2022"),
)


def _definitions():
    definitions = [
        ("text_length", "Number of Unicode code points in the original text."),
        ("utf8_byte_length", "Number of bytes in strict UTF-8 encoding of the original text."),
        ("word_count", f"Number of tokens matched by Unicode regex {WORD_PATTERN}."),
    ]
    for suffix, statistic in (
        ("mean", "Mean"), ("std", "Population standard deviation"),
        ("min", "Minimum"), ("max", "Maximum"),
    ):
        definitions.append((f"word_length_{suffix}", f"{statistic} of original token code-point lengths; zero without tokens."))
    definitions.append(("sentence_count", "Number of segments after splitting on [.!?]+ that contain at least one word token."))
    for suffix, statistic in (
        ("mean", "Mean"), ("std", "Population standard deviation"),
        ("min", "Minimum"), ("max", "Maximum"),
    ):
        definitions.append((f"sentence_word_count_{suffix}", f"{statistic} of word counts in nonempty-word sentence segments; zero without segments."))
    definitions.extend([
        ("unique_word_count", "Number of distinct casefolded word tokens."),
        ("type_token_ratio", "Distinct casefolded word types divided by word count."),
        ("hapax_token_rate", "Tokens whose casefolded type occurs once divided by word count."),
        ("short_token_rate", "Tokens of original code-point length <=3 divided by word count."),
        ("long_token_rate", "Tokens of original code-point length >=7 divided by word count."),
        ("apostrophe_token_rate", "Tokens containing U+0027 or U+2019 divided by word count."),
        ("function_word_rate", "Casefolded tokens in the fixed function-word set divided by word count."),
    ])
    for name, definition in _CHARACTER_CATEGORIES:
        definitions.extend([
            (f"{name}_count", f"Number of individual code points matching {definition}."),
            (f"{name}_rate", f"{name}_count divided by text_length."),
        ])
    for name, characters in _PUNCTUATION:
        codepoints = ", ".join(f"U+{ord(character):04X}" for character in characters)
        definitions.extend([
            (f"{name}_count", f"Number of individual code points matching any of {codepoints}."),
            (f"{name}_rate", f"{name}_count divided by text_length."),
        ])
    return tuple(definitions)


FEATURE_DEFINITIONS = _definitions()
FEATURE_NAMES = tuple(name for name, _ in FEATURE_DEFINITIONS)


def feature_schema():
    """Return a JSON-serializable description independent of any training data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "unicode_database_version": unicodedata.unidata_version,
        "dtype": "float64",
        "word_pattern": WORD_PATTERN,
        "function_words": sorted(FUNCTION_WORDS),
        "zero_denominator_policy": "Return zero for every rate with a zero denominator.",
        "feature_count": len(FEATURE_NAMES),
        "features": [{"name": name, "definition": definition} for name, definition in FEATURE_DEFINITIONS],
    }


def _texts(texts):
    if isinstance(texts, (str, bytes)):
        raise TypeError("Expected an iterable of strings, not one string or bytes object")
    try:
        values = list(texts)
    except TypeError as error:
        raise TypeError("Expected an iterable of strings") from error
    if any(not isinstance(value, str) for value in values):
        raise TypeError("Every stylometry input must be a string")
    return values


def _summary(values):
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    array = np.asarray(values, dtype=np.float64)
    return [float(array.mean()), float(array.std(ddof=0)), float(array.min()), float(array.max())]


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _extract(text):
    length = len(text)
    words = _WORDS.findall(text)
    word_count = len(words)
    lengths = [len(word) for word in words]
    sentences = [len(_WORDS.findall(segment)) for segment in _SENTENCES.split(text)]
    sentences = [count for count in sentences if count]
    types = Counter(word.casefold() for word in words)
    values = [length, len(text.encode("utf-8")), word_count, *_summary(lengths), len(sentences), *_summary(sentences)]
    values.extend([
        len(types),
        _rate(len(types), word_count),
        _rate(sum(count == 1 for count in types.values()), word_count),
        _rate(sum(size <= 3 for size in lengths), word_count),
        _rate(sum(size >= 7 for size in lengths), word_count),
        _rate(sum("'" in word or "\u2019" in word for word in words), word_count),
        _rate(sum(count for word, count in types.items() if word in FUNCTION_WORDS), word_count),
    ])
    categories = Counter(unicodedata.category(character)[0] for character in text)
    counts = [
        sum(character.isspace() for character in text), text.count("\n"), text.count("\t"),
        sum(character.isalpha() for character in text),
        sum(character.isupper() for character in text),
        sum(character.islower() for character in text),
        sum(character.isdigit() for character in text),
        sum(ord(character) > 127 for character in text),
        categories["P"], categories["S"], categories["M"],
    ]
    counts.extend(sum(text.count(character) for character in characters) for _, characters in _PUNCTUATION)
    for count in counts:
        values.extend((count, _rate(count, length)))
    return values


class StylometryExtractor(TransformerMixin, BaseEstimator):
    """Stateless per-text features; ``fit`` never stores corpus statistics."""

    def fit(self, X, y=None):
        _texts(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise TypeError("StylometryExtractor accepts no fit parameters")
        # Materialize once so a generator is not consumed by fit before transform.
        texts = _texts(X)
        return self.fit(texts, y).transform(texts)

    def transform(self, X):
        texts = _texts(X)
        result = np.asarray([_extract(text) for text in texts], dtype=np.float64)
        return result.reshape(len(texts), len(FEATURE_NAMES))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)

    def __sklearn_is_fitted__(self):
        return True


_EXPECTED_CONFIG = {
    "name": "stylometry_logreg_v1",
    "random_seed": 42,
    "n_splits": 5,
    "features": {"schema_version": SCHEMA_VERSION},
    "scaler": {"with_mean": True, "with_std": True},
    "classifier": {"C": 3.0, "max_iter": 1200, "class_weight": "balanced", "solver": "lbfgs", "tol": 1e-4},
}


def validate_stylometry_config(cfg):
    """Reject changes to the preregistered P3 model, including unknown fields."""
    if cfg != _EXPECTED_CONFIG:
        raise ValueError("Config must match the exact preregistered P3 stylometry specification")
    # Equality alone treats True == 1. Do not accept booleans for numeric fields.
    if type(cfg["random_seed"]) is not int or type(cfg["n_splits"]) is not int:
        raise ValueError("P3 random_seed and n_splits must be integers")
    if type(cfg["features"]["schema_version"]) is not int:
        raise ValueError("P3 schema_version must be an integer")
    if any(type(value) is not bool for value in cfg["scaler"].values()):
        raise ValueError("P3 scaler flags must be booleans")
    if type(cfg["classifier"]["max_iter"]) is not int:
        raise ValueError("P3 max_iter must be an integer")


def make_stylometry(cfg):
    """Construct the frozen pipeline; scaling is fit within each CV fold."""
    validate_stylometry_config(cfg)
    classifier = deepcopy(cfg["classifier"])
    classifier["random_state"] = cfg["random_seed"]
    return Pipeline([
        ("features", StylometryExtractor()),
        ("scale", StandardScaler(**cfg["scaler"])),
        ("clf", LogisticRegression(**classifier)),
    ])
