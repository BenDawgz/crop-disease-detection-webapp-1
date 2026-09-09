# Corn/maize, grape, and tomato retraining results

The new model has been trained, tested, and installed as the local runtime model.
Restart the app to load it. The original bundled `best_model.h5` is unchanged.

## What was trained

- 5,193 training images, 1,281 validation images, and 1,176 test images.
- Sources: PlantVillage, PlantDoc, and Imagenette. Attribution and source notes
  are in [training/PUBLIC_DATA.md](training/PUBLIC_DATA.md).
- Disease/healthy categories cover corn/maize, grape, and tomato only.
  Additional categories distinguish non-crop objects and unsupported leaves.
- A new classifier was trained on frozen MobileNetV2 visual features, with
  horizontal flips of training images. Training stopped after 13 epochs;
  validation loss selected the saved weights.
- The original model was evaluated on the same test images for comparison.

## Results and limits

Before the decision thresholds, the new model's highest-scoring category
incorrectly rejected 3 of 776 leaf images, versus 144 for the original. After
the final decision policy, 2 leaf images were rejected and ambiguous predictions
were allowed to remain uncertain.

On 400 Imagenette object test images, the final policy produced no disease
labels and no grouped leaf-recognition messages. **These are not fully
independent negatives:** the ImageNet-pretrained backbone may already have seen
these images. This does not establish an error rate for arbitrary objects,
green objects, or other real-world uploads.

For diseases, the final policy returned labels for 331 of 655 supported-class
test images (50.5% coverage); 323 of those 331 matched the dataset labels.
The laboratory photos dominate this result. **On real-world PlantDoc photos,
only 20 of 115 supported-class images received a disease label, and 16 of those
20 matched (80%, at 17.4% coverage).** Most field photos remain uncertain, and
some confident diagnoses are still wrong. One unsupported leaf also received a
disease label. Further independently labeled field data is needed.

The 95% disease-score threshold was selected using validation data, not the
test set: it was the lowest tested threshold reaching at least 90% accuracy
among at least 20 accepted supported PlantDoc validation images. Model scores
are not calibrated probabilities. Non-crop and unsupported-leaf decisions use
a separate 70% threshold, with a 20-point margin between the leading classes.

All four supplied photos passed a real Flask upload test with the result:
**"A leaf was recognized, but its condition could not be identified reliably."**
No disease-specific treatment was selected. These four images have no verified
disease labels; recognizing their leaf content is not a correct disease diagnosis.

PlantVillage leaf groups were kept within splits. Duplicate/near-duplicate
filtering excluded 68 images, including one similar to the supplied examples.
This reduces leakage but cannot prove all images are independent. PlantDoc's
web labels may be noisy, and the original model's unknown training set may
overlap these public images.

## Files and running the app

- Active model: `instance/models/best_model.h5`
- Matching class map: `instance/class_indices.json`
- Original model: `best_model.h5`
- Downloaded dataset, image manifest, experiment logs, and detailed evaluation:
  `instance/public_retraining/`
- Actual Windows package versions: `instance/public_retraining/environment.json`

From the project folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

If the app is already running, stop it with Ctrl+C first. Startup should report
the runtime model under `instance/models` and an input size of `(224, 224)`.
Custom environment variables such as `MODEL_PATH`, `CLASS_INDICES_PATH`, or
`DISEASE_CONFIDENCE_THRESHOLD` can override this setup.

The downloaded experiment uses a manifest rather than the application upload folder.
Reproduce it with the preparation, training, calibration, and verification
scripts documented in `training/PUBLIC_DATA.md`. Data and runtime model files
under `instance` are intentionally ignored by Git; include them separately
when moving this trained installation to another PC.

Tomato spotted wilt and leaf-miner damage were not added as disease categories.
Healthy tomato and tomato mosaic virus were added from labeled public data.
