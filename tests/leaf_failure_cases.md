# User-supplied leaf failure cases

Evaluated on 2026-09-08 using the bundled `best_model.h5`, bundled class mapping,
TensorFlow 2.21.0, RGB images resized to 256 x 256 with the application's default
nearest-neighbor interpolation, and pixel values divided by 255. No retraining
or test-image augmentation was performed. Files remain in the user's Downloads
folder and are not included in the repository.

All four images visibly contain leaves. Disease labels have not been verified.
Model scores below are softmax scores, not calibrated accuracy estimates.

| File | Highest-scoring class | Score | Current decision |
| --- | --- | --- | --- |
| images (1).jpg | Corn gray leaf spot | 24.54% | Uncertain image |
| images (3).jpg | Not_A_Crop | 85.27% | image not supported |
| images (4).jpg | Not_A_Crop | 80.57% | image not supported |
| tomato-disease-11-TSWV.jpg | Corn gray leaf spot | 48.09% | Uncertain image |

The second and third examples demonstrate remaining false non-crop predictions
in the trained model, even after removing the color gate. The first and fourth
demonstrate successful abstention from weak disease predictions, not successful
leaf or disease identification. None of these cases proves disease accuracy.

The class mapping has no tomato spotted wilt class. The fourth filename suggests
that condition but is not ground truth. A confidence threshold cannot add an
absent disease category or reliably detect all unfamiliar images.

The rejection message now explains that unsupported images can still contain
leaves. This is a reporting correction, not a model-accuracy fix.

Next model work requires verified examples of supported and unsupported leaf
conditions, along with real non-leaf objects (including green objects). Evaluate
leaf recognition separately from disease classification. Keep these four cases
as held-out checks; do not train on them and report their reuse as independent
validation. Measure false rejection of leaves and false acceptance of non-leaves
on a larger independent set before selecting thresholds or replacing the model.
