"""Conservative decisions over the model's complete class distribution."""

import numpy as np


def classify_scores(scores, class_names, confidence_threshold=0.70,
                    margin_threshold=0.20, non_crop_class="Not_A_Crop",
                    disease_confidence_threshold=None):
    """Return (label or None for uncertainty, confidence percentage).

    Compare against *all* classes, including the negative class. Never promote
    a runner-up crop when the model prefers a non-crop image. Color alone is
    not evidence of a leaf, and is deliberately not part of this decision.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or len(scores) != len(class_names)
            or len(scores) < 2 or not np.all(np.isfinite(scores))
            or len(set(class_names)) != len(class_names)
            or non_crop_class not in class_names):
        return None, 0.0

    if np.all((scores >= 0) & (scores <= 1)) and np.isclose(scores.sum(), 1.0):
        probs = scores / scores.sum()
    else:
        # Stable softmax for models whose output layer emits logits.
        weights = np.exp(scores - np.max(scores))
        probs = weights / weights.sum()

    ranked = np.argsort(probs)
    best, runner_up = int(ranked[-1]), int(ranked[-2])
    confidence = float(probs[best])
    margin = confidence - float(probs[runner_up])
    threshold = confidence_threshold
    if (disease_confidence_threshold is not None
            and class_names[best] not in (non_crop_class, 'Unsupported_Leaf')):
        threshold = disease_confidence_threshold
    if confidence < threshold or margin < margin_threshold:
        return None, confidence * 100.0
    return class_names[best], confidence * 100.0


def has_leaf_evidence(scores, class_names):
    """Conservative grouped leaf evidence for the public-data classifier only.

    This is a model decision, not a calibrated probability or disease diagnosis.
    Do not apply it to the old model, which lacked an unsupported-leaf class.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if ('Unsupported_Leaf' not in class_names or 'Not_A_Crop' not in class_names
            or len(set(class_names)) != len(class_names)
            or scores.ndim != 1 or len(scores) != len(class_names)
            or not np.all(np.isfinite(scores)) or np.any(scores < 0)
            or np.any(scores > 1) or not np.isclose(scores.sum(), 1.0)):
        return False
    return float(scores[class_names.index('Not_A_Crop')]) <= 0.01
