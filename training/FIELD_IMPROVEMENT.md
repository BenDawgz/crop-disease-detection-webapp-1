# Field-photo improvement experiment

This experiment creates candidates in `instance/field_improvement`. The refined
runtime model remains installed until a separately evaluated replacement is
selected. A prepared script or successful training run is not evidence of better
disease detection; consult the generated evaluations.

## Data and provenance

- PlantWild v2, Tianqi Wei and coauthors, *Benchmarking In-the-Wild Multimodal
  Plant Disease Recognition and A Versatile Baseline* (ACM Multimedia 2024).
  The authors report agricultural-expert refinement of v2:
  https://tqwei05.github.io/PlantWild/ . Dataset license: CC BY-NC-ND 4.0;
  this is a local research experiment, not a redistribution or a blanket grant
  for commercial deployment. Pinned author-hosted release:
  https://huggingface.co/datasets/uqtwei2/PlantWild/tree/527a72eb8f00c95e41698bb09d982c7d4625dce3 .
  Archive SHA256: `5c3fd0c6ecf9a346c9d2dd03ff47769c71e581eb1170a68477d718ab43cfee5b`.
- PlantDoc bounding boxes, Davinder Singh and coauthors (CoDS-COMAD 2020),
  CC BY 4.0:
  https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/tree/4730a233a555b30ee98e0879c63ad25d82407455 .
  The original image coordinates supervise a dominant-leaf locator. Boxes do
  not represent segmentation masks. Invalid annotations and conflicting exact
  duplicate partitions are excluded and logged.
- Existing PlantVillage, PlantDoc classification and Imagenette data retain
  their attribution and limitations in `PUBLIC_DATA.md`.

PlantSeg outlines *diseased parts*, not whole leaves. Its masks are consequently
not used as whole-leaf annotations. PlantWild's broader plant-disease photos can
include fruit or stems; its disease labels do not establish leaf presence in
every image. Per-image review is still required before claiming a leaf-only set.

## Workflow on Windows

Run from the project directory using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe training/download_field_sources.py
.\.venv\Scripts\python.exe training/prepare_field_data.py
.\.venv\Scripts\python.exe training/apply_field_review.py
.\.venv\Scripts\python.exe training/prepare_leaf_boxes.py
.\.venv\Scripts\python.exe training/train_leaf_locator.py --variant final
.\.venv\Scripts\python.exe training/finetune_field_model.py
.\.venv\Scripts\python.exe training/evaluate_two_stage.py --locator-variant final
.\.venv\Scripts\python.exe training/verify_field_candidate.py
.\.venv\Scripts\python.exe training/verify_field_candidate.py --with-locator
```

The downloader can resume an interrupted transfer after validating the server's
byte-range response, and verifies the complete 1.6 GB archive before use. Classification
preparation maps exact known conditions, adds grape downy mildew and leafroll,
and skips ambiguous `grape leaf spot` and corn smut rather than guessing a
leaf-disease mapping. It keeps all legacy held-out rows, caps lab training at 120
unique groups per class, and retains existing non-leaf/unsupported/healthy data.
Ambiguous legacy groups crossing partitions are excluded conservatively.
New images similar to any legacy image or reserved user example are excluded. Connected components of
near duplicates stay together; only one view enters a new partition.

All 1,586 retained PlantWild images were inspected on contact sheets for subject
and suitability. `field_visual_review.json` records decisions against exact
archive members and pixel hashes. The review retains 1,205 leaf photos, assigns
178 fruit/stem subjects to `Not_A_Crop`, and excludes 203 mixed/composite or
unsuitable images. This is AI visual review, not independent expert confirmation
of disease. Dataset-author disease labels are retained for leaf photos. The
review script fails if source membership or reviewed image pixels change.

The classifier warms its expanded output head, then unfreezes MobileNetV2 from
block 13 onward at a lower learning rate. Batch-normalization statistics remain
fixed. Training-only brightness and horizontal flips vary appearance. Validation
loss selects the checkpoint. Disease decisions are compared with the refined
baseline at the same 0.95 score threshold. Scores are not calibrated diagnosis
probabilities. The baseline cannot name newly added classes, which is visible in
the per-class results.

The dominant-leaf locator is trained separately with original boxes and negative
images. Its gate requires at least 70% of validation leaves localized with IoU
at least 0.5, with at most 2% negative false detections. These are development
criteria, not a certification of field accuracy. `evaluate_two_stage.py` measures
the effect of predicted crops on diagnosis, including the loss of coverage when
the locator abstains. Cropping is not assumed to improve diagnosis automatically.

## Runtime integration

The app supports `LEAF_LOCATOR_PATH` and `LEAF_LOCATOR_THRESHOLD`. These are unset
by default. Enabling a locator makes it run before disease classification; an
uncertain or malformed box produces an uncertain result with no treatment.
The crop includes 8% context. This version selects one dominant leaf, so users
should still photograph one leaf at a time.

Do not enable a candidate solely because it trained successfully. Review both
the locator results and the whole-image versus two-stage disease results first.
The `.keras` field classifier is a candidate artifact, not an automatically
installed replacement for the app's runtime `.h5` file.

## Review and genuinely independent testing

`field_review_template.json` records crop, condition, leaf boxes, reviewer and
diagnostic evidence alongside plant, farm and photo-session identifiers. The four
user failure images are listed in `instance/field_improvement/user_examples_for_review.json`
with disease labels unassigned; a filename is not verified ground truth.

Collect healthy leaves, disease stages, pest damage, confirmed nutrient problems,
fruit/stems, soil, hands and non-leaf objects under varied lighting and distances.
Use reviewers with relevant plant-pathology expertise. Keep uncertain diagnoses
pending rather than assigning a disease from appearance alone.

Reserve entire farms/sessions/plants for final testing before training. Images,
crops, augmented copies and repeat views of those plants must remain outside
training and validation. Freeze the test manifest and evaluate only the selected
candidate, reporting disease precision/recall, abstention coverage, leaf false
rejections and non-leaf false acceptance separately.

Public web datasets lack verified plant/session identifiers. Perceptual filtering
reduces overlap but cannot turn them into a genuinely independent prospective
field trial. The previously used public tests remain regression benchmarks.
Spotted wilt, leaf-miner damage, nutrient-deficiency categories and a reviewed
local field test remain collection gaps, not completed training categories.
