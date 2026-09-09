# Retraining with the uploaded dataset — 2026-09-08

**The refined candidate was activated at the user's request on 2026-09-08.**
It improves performance on the supplied dataset's style of images without
demonstrating an improvement in the reported field-photo problem. Restart the
app to load the new runtime model. The evaluation limitations below still apply.

## Data and training

- Found 29,167 files in 16 maize/corn, grape, and tomato classes.
- Retained public field photos, non-leaf objects, unsupported leaves, healthy
  tomato, and tomato mosaic examples to preserve the prior coverage.
- Used 23,609 training, 2,472 validation, and 2,390 test images.
- Kept related leaf groups in one split and preserved the earlier public test
  groups. Scored only one newly held-out view per local leaf rather than its
  many rotated/flipped copies. All local images decoded successfully.
- First trained a new classifier over frozen MobileNetV2 features. It failed
  the field-validation acceptance criterion.
- Then refined the installed classifier over the same features with a smaller
  learning rate. Selected epoch 16 using validation accuracy subject to the
  field-validation criterion. Disease threshold remained 95% model score.

## Comparison with the currently installed model

| Measurement | Current model | Refined candidate |
| --- | ---: | ---: |
| Correct top-ranked labels on 1,214 held-out user-dataset images | 87.2% | 93.1% |
| User-dataset images given a disease/healthy label | 678 / 1,214 | 801 / 1,214 |
| Correct among those accepted user-dataset labels | 98.7% | 99.6% |
| Field images given a disease/healthy label | 20 / 115 | 18 / 115 |
| Correct among those accepted field labels | 16 / 20 | 14 / 18 |
| Non-leaf test images wrongly given a disease label | 0 / 400 | 1 / 400 |

The candidate gives more answers on laboratory-style photos. It still leaves
most real-world photos uncertain, and its small accepted field subset did not
improve. These field differences are too small to establish a statistically
reliable ranking; they do not demonstrate a fix for the reported problem.

The dataset has no spotted-wilt or leaf-miner category. More copies or rotations
of existing classes cannot teach those missing diagnoses. Recognition of a leaf
is distinct from correctly identifying its condition.

## Artifacts

- Refined candidate: `instance/local_retraining/refined_candidate.h5`
- Matching class map: `instance/local_retraining/class_indices.json`
- Detailed comparison: `instance/local_retraining/refined_runtime_comparison.json`
- Epoch/validation selection: `instance/local_retraining/refinement_selection.json`
- Manifest and data counts: `instance/local_retraining/`
- Reproduction instructions: `training/LOCAL_DATA.md`

Your source images and original bundled model were preserved. The previous
runtime model and class map were backed up under
`instance/local_retraining/runtime_backup_20260908_063507`. The refined model
and matching class map are now installed under `instance/models/best_model.h5`
and `instance/class_indices.json`. Restart the app to load them.

The public test set is reused as a regression benchmark, and the original
dataset's leaf identities cannot be independently guaranteed from filenames
alone. Imagenette may overlap the backbone's ImageNet pretraining. These
results are not an independent field-deployment accuracy guarantee. The needed
next data is independently labeled field photos and the missing disease/pest
categories, not additional augmentations of the same laboratory photos.
