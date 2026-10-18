import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

from .labels import LABELS


def make_baseline(cfg: dict) -> Pipeline:
    word = dict(cfg["word"])
    char = dict(cfg["char"])
    word["ngram_range"] = tuple(word["ngram_range"])
    char["ngram_range"] = tuple(char["ngram_range"])
    # TfidfVectorizer fitted INSIDE each fold, never globally.
    features = FeatureUnion([
        ("word", TfidfVectorizer(**word)),
        ("char", TfidfVectorizer(**char)),
    ])
    classifier = dict(cfg["classifier"])
    classifier.setdefault("random_state", cfg["random_seed"])
    clf = LogisticRegression(**classifier)
    return Pipeline([("features", features), ("clf", clf)])


def ordered_probabilities(model, texts) -> np.ndarray:
    """Return validated probabilities in the competition's fixed label order."""
    classes = list(model.classes_)
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError("Model must predict exactly the 18 legal labels")
    probabilities = np.asarray(model.predict_proba(texts), dtype=np.float64)
    if probabilities.shape != (len(texts), len(LABELS)):
        raise ValueError("Unexpected probability matrix shape")
    probabilities = probabilities[:, [classes.index(label) for label in LABELS]]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Probabilities must sum to one")
    return probabilities
