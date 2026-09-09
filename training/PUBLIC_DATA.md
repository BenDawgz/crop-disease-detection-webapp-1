# Public-data retraining experiment

The preparation and training scripts create a candidate under
`instance/public_retraining`. They do not overwrite or activate the original
model. Downloaded images, manifests, dataset notices, features, and reports stay
in that ignored local directory.

## Sources and attribution

- **PlantVillage**, Sharada P. Mohanty, David P. Hughes, Marcel Salathé (2016),
  *Using Deep Learning for Image-Based Plant Disease Detection*.
  https://github.com/spMohanty/PlantVillage-Dataset
  DOI: 10.3389/fpls.2016.01419.
  The authors' dataset card declares **CC BY-SA 3.0**:
  https://huggingface.co/datasets/mohanty/PlantVillage.
  This experiment samples corn, grape, and tomato classes, preserving the
  original leaf-group metadata when selecting separate data splits.
- **PlantDoc**, Davinder Singh et al. (2020), *PlantDoc: A Dataset for Visual
  Plant Disease Detection*. https://github.com/pratikkayal/PlantDoc-Dataset.
  Repository dataset license: **CC BY 4.0**, retained with the downloaded data.
  Mapped crop conditions supplement the laboratory photos; other plant species
  form an `Unsupported_Leaf` class. This is not an exhaustive unknown-disease set.
- **Imagenette**, Jeremy Howard / fast.ai.
  https://github.com/fastai/imagenette. Subset of ImageNet with ten object classes.
  Repository license: **Apache 2.0**; original image rights are not established
  by that code license. This local experimental dataset is not being published
  or offered with a blanket image redistribution license.

Images are RGB-converted and JPEG-reencoded for the local experiment. Model
inputs are resized to 224 x 224. The per-image manifest retains URLs and source
identifiers; GitHub dataset revisions are saved. Review source image rights
before redistributing the downloaded dataset.

## Reproduce on Windows

From the project folder, using the installed Python environment:

```powershell
.\.venv\Scripts\python.exe training/prepare_public_data.py
.\.venv\Scripts\python.exe training/train_public_model.py
.\.venv\Scripts\python.exe training/calibrate_public_model.py
.\.venv\Scripts\python.exe training/verify_public_candidate.py
```

The first command downloads a bounded subset and filters duplicate and visually
similar images. The second trains a classifier over frozen ImageNet-pretrained
MobileNetV2 features. Horizontal flips augment training only. Validation loss
selects the checkpoint; the test set and four supplied failure examples are not
used for fitting or early stopping. The original bundled disease model is
evaluated on the same test images.

The third command selects a disease-score threshold from validation data and
reports the fixed policy on test images. The fourth exercises the real Flask
upload route with the four supplied photos, using an isolated database. These
scripts do not activate a newly generated candidate automatically. The completed
run and installed runtime model are documented in `RETRAINING_RESULTS.md`.

The exported candidate accepts the app's existing RGB [0, 1] inputs. The extra
MobileNet normalization is embedded in the saved model. The candidate and its
class mapping must always be deployed together.

## Interpretation

This is a pilot, not a guarantee of field accuracy. Report source-specific
metrics, disease coverage/accuracy, false leaf rejection, and false acceptance
of non-leaf objects. Keep uncertain results instead of forcing a disease label.
Imagenette images may have been used in the backbone's ImageNet pretraining,
so its negative test scores are not an independent pretraining benchmark.
The original crop model's training data is unknown and may also overlap public
images. Hash filtering reduces duplication but cannot establish complete image
independence. PlantDoc's web labels may be noisy.

Tomato mosaic and healthy tomato are additional categories. Tomato spotted
wilt and leaf-miner damage are not added: this experiment has no verified
training category for them. The four user images are used only to check leaf
rejection, not to claim a verified disease diagnosis.
