# Uploaded-dataset experiment

Source: `training/dataset`, supplied by the user. The 29,167 files have 16
corn/maize, grape, and tomato labels and include rotated/flipped copies with
PlantVillage-style names. The source files are read without modification.

`prepare_local_data.py` canonicalizes augmentation suffixes and uses the retained
PlantVillage leaf map. It keeps the previous public held-out groups out of
training, selects one view for newly held-out local leaf groups, checks decoding,
and excludes exact duplicate pixels and groups similar to the supplied failure
examples. This does not prove that all near-duplicates or related leaves have
been identified. Existing PlantDoc, Imagenette, healthy tomato, and tomato mosaic
examples are retained to preserve field-photo and negative coverage.

The resulting manifest has 23,609 training, 2,472 validation, and 2,390 test
images. All class/split counts and exclusions are saved in
`instance/local_retraining`. Downloaded source attribution remains in
`training/PUBLIC_DATA.md`.

Run from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe training/prepare_local_data.py
$env:RETRAIN_EXPERIMENT_DIR = "$PWD\instance\local_retraining"
$env:RETRAIN_FEATURE_FLIPS = "0"
$env:RETRAIN_FIELD_WEIGHT = "3"
.\.venv\Scripts\python.exe training/train_public_model.py
.\.venv\Scripts\python.exe training/refine_local_head.py
$env:RETRAIN_CANDIDATE_VARIANT = "refined"
.\.venv\Scripts\python.exe training/compare_local_candidate.py
```

The training images already include augmentation, so no additional flipped
feature pass is used. PlantDoc examples receive 3x source weight during fitting
and validation-loss selection, to reduce domination by laboratory-style photos.
The visual backbone remains frozen; a classifier is retrained on its features.
The uploaded files are not proof of coverage of spotted wilt or leaf-miner
damage: neither is a labeled category here.

The scripts produce and evaluate a candidate; they do not overwrite the runtime
model. Public test images are now a reused regression benchmark, not a fresh
independent final test. Imagenette's overlap with ImageNet pretraining also
limits interpretation of negative-example scores. Report field performance and
abstention coverage separately from overall accuracy before activating a model.

The first newly initialized head failed the field-validation criterion. The
refinement starts from the installed classifier at learning rate 0.0001 and
selects an epoch using validation data only: supported-class accuracy must
improve while at least 20 accepted supported PlantDoc validation predictions
reach 90% accuracy. Epoch 16 was selected from 20 epochs. The selected disease
score threshold remained 0.95. Results are in `LOCAL_RETRAINING_RESULTS.md`.
