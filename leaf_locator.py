"""Dominant-leaf localization utilities, shared by training and inference.

Coordinates are normalized xmin, ymin, xmax, ymax. A locator's object score
does not establish a disease diagnosis. Models must be validated before use.
"""
import numpy as np


def box_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    union = np.prod(np.maximum(0, a[2:] - a[:2])) + np.prod(np.maximum(0, b[2:] - b[:2])) - intersection
    return float(intersection / union) if union > 0 else 0.0


def valid_box(box):
    box = np.asarray(box, dtype=float)
    return (box.shape == (4,) and np.all(np.isfinite(box)) and
            np.all(box >= 0) and np.all(box <= 1) and
            box[2] - box[0] >= .05 and box[3] - box[1] >= .05)


def crop_leaf(image, box, padding=.08):
    """Crop one predicted leaf with context; reject invalid boxes."""
    if not valid_box(box):
        raise ValueError('Invalid leaf bounding box')
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    w, h = image.size
    return image.crop((int(max(0, x1 - dx) * w), int(max(0, y1 - dy) * h),
                       int(np.ceil(min(1, x2 + dx) * w)), int(np.ceil(min(1, y2 + dy) * h))))


def locate_leaf(model, image, threshold=.9):
    """Return a dominant-leaf box or None, never force a crop on low confidence."""
    from PIL import Image
    pixels = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32) / 255
    result = model(pixels[None], training=False)
    score = float(np.asarray(result['leaf'])[0, 0])
    box = np.asarray(result['box'])[0]
    if not np.isfinite(score) or score < threshold or not valid_box(box):
        return None
    return box.tolist()
