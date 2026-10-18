from name_that_llm.metrics import classification_report
from name_that_llm.labels import LABELS
import pytest


def test_perfect_all_labels():
    report=classification_report(LABELS,LABELS)
    assert report['balanced_accuracy']==1.0
    assert len(report['class_recall'])==18


def test_balanced_accuracy_weights_classes_not_rows():
    # A large first class is wholly wrong; every other class is perfect.
    truth = [LABELS[0]] * 100 + LABELS[1:]
    predictions = [LABELS[1]] * 100 + LABELS[1:]
    report = classification_report(truth, predictions)
    assert report['balanced_accuracy'] == pytest.approx(17 / 18)
    assert report['confusion_matrix'][0][1] == 100
    assert sum(map(sum, report['confusion_matrix'])) == 117


def test_missing_or_illegal_evaluation_label_rejected():
    with pytest.raises(ValueError, match='missing'):
        classification_report(LABELS[:-1], LABELS[:-1])
    with pytest.raises(ValueError, match='Illegal'):
        classification_report(LABELS, ['invalid'] + LABELS[1:])
