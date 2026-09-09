# Field-data experiment results — 8 September 2026

The field candidate was trained and evaluated. The installed refined model is
unchanged. The candidate improves field-photo performance but loses some accuracy
on the uploaded dataset and still leaves most field conditions uncertain.

## Completed work

- Downloaded and checksum-verified the author-hosted PlantWild v2 archive.
- Reviewed 1,586 added images for suitability: retained 1,205 leaf photos,
  assigned 178 fruit/stem subjects to the non-leaf category, and excluded 203
  mixed/composite or unsuitable images. This was AI visual screening, not an
  independent plant-pathologist review of diseases. Leaf disease labels remain
  those supplied by the dataset authors.
- Removed exact/perceptual overlaps and ambiguous cross-partition groups.
- Trained on 6,159 images; used 2,673 for validation and 2,591 for testing.
- Added grape downy mildew and grape leafroll disease, preserving existing
  healthy, disease, unsupported-leaf and non-leaf categories: 22 outputs total.
- Warmed the expanded classifier for two epochs, then fine-tuned 26 MobileNetV2
  visual layers for up to four epochs, keeping BatchNorm statistics fixed.
  Validation loss selected the second deeper-training epoch.
- Trained a separate dominant-leaf locator and implemented optional crop-first
  inference. It remains disabled because its localization validation gate failed.
- Passed 24 regression tests and real Flask upload checks with and without the
  candidate locator, using isolated storage and a test database.

## Comparison on diseases both models support

Both models use a 0.95 disease-score threshold and a 0.20 score margin. Top-choice
accuracy scores the highest-ranked label even when the app abstains. Accepted
accuracy scores only the disease labels the app would actually issue.

| Held-out source | Measure | Refined runtime | Field candidate |
|---|---|---:|---:|
| Added PlantWild field photos | Top-choice accuracy on 141 shared-category leaf photos | 43.3% | 57.4% |
| Added PlantWild field photos | Accepted labels, correct / issued | 21 / 26 | 31 / 35 |
| Added PlantWild field photos | Fraction of shared-category leaves given a label | 18.4% | 24.8% |
| Existing PlantDoc field benchmark | Top-choice accuracy on 111 supported leaf photos | 38.7% | 48.6% |
| Existing PlantDoc field benchmark | Accepted labels, correct / issued | 14 / 18 | 18 / 22 |
| Uploaded dataset, 1,214 test images | Top-choice accuracy | 93.1% | 90.0% |
| Uploaded dataset | Accepted labels, correct / issued | 798 / 801 | 777 / 785 |

Across all 178 added field leaf photos, including the two new categories, the
candidate's top-choice accuracy is 56.7%, compared with 34.3% for the refined
model, which cannot name the new categories. The candidate issues 42 disease
labels, of which 37 match the dataset labels. Neither model issued a disease
label for the 27 added fruit/stem test subjects at this threshold. This small
negative set is not proof of general non-leaf reliability.

The candidate falsely rejected five PlantDoc leaves and two PlantWild leaves
as non-leaf, compared with two and one respectively for the refined model.
These regressions, the low field coverage and the uploaded-dataset regression
are reasons to retain the refined runtime while reviewing this candidate.

New-category results are preliminary: grape downy mildew has 26 test photos and
four accepted correct diagnoses; grape leafroll has 11 test photos and two
accepted correct diagnoses. Leafroll has only three validation examples.

## Locator and upload behavior

The corrected locator finds a leaf in 97.5% of its 238 positive test images,
but only 67.6% are localized with bounding-box overlap of at least 0.5. On its
validation set that localization measure is below the predeclared 70% target.
Its public benchmark is reused, and its Imagenette negatives may overlap
ImageNet pretraining. It must not be interpreted as a new independent field test.

All four supplied failure photos load through the actual upload route. With
either whole-image or crop-first candidate inference, all four remain
"leaf recognized; condition uncertain" and receive no disease-specific treatment.
Their disease ground truth has not been independently verified.

On all 178 added field leaf photos, cropping reduced issued disease labels
from 42 to 28, and correct issued labels from 37 to 26. Accuracy among issued
labels increased from 88.1% to 92.9%, but coverage fell from 23.6% to 15.7%.
Cropping therefore does not solve the frequent-uncertainty problem. The locator
remains disabled. Failed localization is counted as unclassified in the report.

## Remaining data needs

Tomato spotted wilt, leaf-miner damage and nutrient deficiencies remain missing
verified categories. Healthy field examples and local non-leaf lookalikes also
need broader coverage. Public web datasets do not provide verified plant/farm/
photo-session identifiers, so this experiment is not a genuinely independent
prospective field trial. Use the supplied review template to collect that set.

## Artifacts and reproduction

- Candidate: `instance/field_improvement/field_candidate.keras`
- Matching map: `instance/field_improvement/class_indices.json`
- Per-disease precision, recall and confusion counts: `instance/field_improvement/field_evaluation.json`
- Shared-category comparison: `instance/field_improvement/common_class_comparison.json`
- Locator: `instance/field_improvement/final_leaf_locator.keras`
- Locator evaluation: `instance/field_improvement/final_locator_evaluation.json`
- Whole-image versus crop-first comparison: `instance/field_improvement/final_two_stage_evaluation.json`
- Review decisions: `training/field_visual_review.json`
- Collection template: `training/field_review_template.json`
- Windows commands and source attribution: `training/FIELD_IMPROVEMENT.md`

Candidate SHA256: `aa6ecb658146d72416426a18a34f862af4e97ffd1d9e592fbb3d4c58ffafc25b`.
Unchanged refined runtime SHA256: `7f249c91878aaa7510df12d812090886e81ef185cc57142a23fef5f64759d4fe`.
