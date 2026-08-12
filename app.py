import tensorflow as tf
from flask import Flask, render_template, request, Response, flash, redirect
import cv2
import os
from werkzeug.utils import secure_filename
from random import randint
from tensorflow.keras.models import load_model
import numpy as np

# Globals
global capture, switch, filename
capture = 0
switch = 0
filename = ""

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/shots'
app.secret_key = 'cropdisease'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure shots folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Camera - only initialize if not in production (Render has no webcam)
# Camera disabled
camera = None

# Random name for captured image
variable_name = str(randint(0, 100))
size = len(variable_name)

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global capture
    if camera is None:
        return
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                p = os.path.sep.join([UPLOAD_FOLDER, f"{variable_name}.png"])
                cv2.imwrite(p, frame)
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/input", methods=['GET', 'POST'])
def input():
    return render_template("input.html")

@app.route('/video')
def video():
    if camera is None:
        # Return a 1x1 transparent pixel instead of hanging
        return Response(b'', status=204)

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, capture
    if request.method == 'POST':
        if request.form.get('click') == 'Capture Image':
            capture = 1
            # After capture, jump to display using the camera file
            return redirect("/upload")  # handled as GET below
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                if camera is not None:
                    camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        return redirect("/upload")
    return render_template('display.html')

# ------------------------------
# Model and Classes
# ------------------------------
MODEL_PATH = "best_model.h5"
CONFIDENCE_THRESHOLD = 0.40  # 40%

NEW_CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

TREATMENT_DICT = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use fungicides like mancozeb and rotate crops.",
    "Corn_(maize)___Common_rust_": "Apply fungicides at early infection stage and plant resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and apply fungicides.",
    "Corn_(maize)___healthy": "No disease detected. Continue good practices.",
    "Grape___Black_rot": "Remove infected fruits, prune vines, and apply fungicides.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood and avoid water stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and prune infected leaves.",
    "Grape___healthy": "No disease detected. Maintain good vineyard hygiene.",
    "Tomato___Bacterial_spot": "Use copper-based sprays and remove infected leaves.",
    "Tomato___Early_blight": "Apply fungicides and rotate crops annually.",
    "Tomato___Late_blight": "Remove infected plants and apply fungicides promptly.",
    "Tomato___Leaf_Mold": "Increase ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove affected leaves and use fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or neem oil.",
    "Tomato___Target_Spot": "Apply fungicides and ensure proper plant spacing.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use insecticides to control whiteflies, remove infected plants, and use resistant varieties."
}

def _infer_model_img_size(model, fallback=(224, 224)):
    """
    Try to read (H, W) from the loaded model.
    Fallback to 224x224 (matches the 36,864 Flatten -> 12x12x256 signature).
    """
    try:
        ishape = model.input_shape
        if isinstance(ishape, (list, tuple)):
            if isinstance(ishape[0], (list, tuple)):
                ishape = ishape[0]
        h, w = ishape[1], ishape[2]
        if isinstance(h, int) and isinstance(w, int):
            return (h, w)
    except Exception:
        pass
    return fallback

try:
    MODEL = load_model(MODEL_PATH)
    MODEL_IMG_SIZE = _infer_model_img_size(MODEL, fallback=(224, 224))
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Inferred model input size: {MODEL_IMG_SIZE}")
except Exception as e:
    MODEL = None
    MODEL_IMG_SIZE = (224, 224)
    print(f"Error loading model: {e}")

def _safe_softmax(x):
    # If last layer already softmax, values will sum ~1 in [0,1]; otherwise apply softmax.
    x = np.asarray(x).astype("float32")
    s = x.sum()
    if np.any(x > 1.0001) or s <= 0.0 or s > 1.0001:
        return tf.nn.softmax(x).numpy()
    return x

def is_leaf_image(image_path, min_plant_ratio=0.08):
    """
    Check if an image likely contains a plant/leaf by analyzing color distribution.
    Uses HSV color space to detect green, yellow-green, and brown (diseased leaf) hues.
    Returns (is_leaf: bool, reason: str)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read the image file."

        # Resize for faster processing
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # Define plant-like color ranges in HSV
        # Green hues (healthy leaves): H=25-90
        green_mask = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))
        # Yellow-brown hues (diseased/dying leaves): H=10-25
        yellow_brown_mask = cv2.inRange(hsv, (10, 30, 30), (25, 255, 255))
        # Dark brown/necrotic (severely diseased): H=0-10 with low-mid saturation
        brown_mask = cv2.inRange(hsv, (0, 20, 20), (10, 200, 180))

        # Combine all plant-related colors
        plant_mask = green_mask | yellow_brown_mask | brown_mask
        plant_pixels = cv2.countNonZero(plant_mask)
        plant_ratio = plant_pixels / total_pixels

        if plant_ratio >= min_plant_ratio:
            return True, f"Plant content detected ({plant_ratio:.0%})."

        return False, (
            "This doesn't appear to be a leaf image. "
            "Please upload a clear photo of a crop leaf (corn, grape, or tomato)."
        )
    except Exception as e:
        # If validation fails, allow the image through (fail-open)
        return True, f"Validation skipped: {e}"

def processing(fname):
    global MODEL
    if MODEL is None:
        return "Model not loaded", 0.0, None

    image_path = os.path.join(UPLOAD_FOLDER, fname)
    if not os.path.exists(image_path):
        return "No image", 0.0, None

    # Step 1: Check if the image looks like a leaf
    is_leaf, reason = is_leaf_image(image_path)
    if not is_leaf:
        return reason, 0.0, None

    # Step 2: Run the model
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=MODEL_IMG_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0).astype("float32") / 255.0

    preds = MODEL.predict(input_arr)
    scores = preds[0]
    probs = _safe_softmax(scores)
    idx = int(np.argmax(probs))
    label = NEW_CLASS_NAMES[idx] if idx < len(NEW_CLASS_NAMES) else f"Class {idx}"
    confidence = float(probs[idx]) * 100.0

    if confidence < (CONFIDENCE_THRESHOLD * 100.0):
        return "Uncertain - Image may not be a crop leaf", confidence, None

    treatment = TREATMENT_DICT.get(label, "No treatment info available")
    return label, round(confidence, 2), treatment

# ---------- Upload / Display ----------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    GET  -> show last camera capture (variable_name.png) if present
    POST -> handle file upload
    """
    global filename

    if request.method == 'GET':
        # Camera path
        cam_file = f"{variable_name}.png"
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_file)
        if os.path.exists(cam_path):
            filename = cam_file
            label, confidence, treatment = processing(filename)
            return render_template('display.html',
                                   variable_name=filename,
                                   label=label,
                                   confidence=confidence,
                                   treatment=treatment)
        # no camera file -> back to input
        flash("No captured image found. Please upload an image.")
        return redirect('/input')

    # POST (file upload)
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        flash("Image successfully uploaded.")
        label, confidence, treatment = processing(filename)
        return render_template('display.html',
                               variable_name=filename,
                               label=label,
                               confidence=confidence,
                               treatment=treatment)
    else:
        flash("Allowed image types are - png, jpg, jpeg, gif.")
        return redirect('/input')

@app.route('/display')
def display_image():
    # Use the last known filename
    if not filename:
        flash("No image to display.")
        return redirect('/input')
    label, confidence, treatment = processing(filename)
    return render_template('display.html',
                           variable_name=filename,
                           label=label,
                           confidence=confidence,
                           treatment=treatment)

if __name__ == "__main__":
    app.run(debug=True)