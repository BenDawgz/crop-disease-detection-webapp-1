import tensorflow as tf
from flask import Flask, render_template, request, Response, flash, redirect, session, url_for
import cv2
import json
import os
from pathlib import Path
from uuid import uuid4
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np

# Globals
global switch
switch = 0

# Flask App Setup
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'shots'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-cropdisease-change-me')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ENABLE_CAMERA = os.environ.get('ENABLE_CAMERA', '').lower() in {'1', 'true', 'yes', 'on'}

# Ensure shots folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Camera is disabled by default because hosted environments usually have no webcam.
camera = None

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def unique_upload_name(original_name):
    safe_name = secure_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem[:50] or 'upload'
    return f"{uuid4().hex}_{stem}{suffix}"

def upload_path(fname):
    safe_name = secure_filename(fname or '')
    if not safe_name or safe_name != fname:
        return None
    path = UPLOAD_FOLDER / safe_name
    try:
        path.resolve().relative_to(UPLOAD_FOLDER.resolve())
    except ValueError:
        return None
    return path

def generate_frames():
    if camera is None:
        return
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
        if not ret:
            continue
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
        return Response(b'', status=204)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if not ENABLE_CAMERA:
        flash("Camera capture is disabled. Please upload an image instead.")
        return redirect('/input')

    if request.method == 'POST':
        if request.form.get('click') == 'Capture Image':
            if camera is None:
                camera = cv2.VideoCapture(0)
            success, frame = camera.read()
            if not success:
                flash("Could not capture an image from the camera.")
                return redirect('/input')
            captured_name = f"camera_{uuid4().hex}.png"
            cv2.imwrite(str(UPLOAD_FOLDER / captured_name), frame)
            session['last_filename'] = captured_name
            return redirect(url_for('display_image', filename=captured_name))
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                if camera is not None:
                    camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        return redirect('/input')
    return redirect('/input')

# ------------------------------
# Model and Classes
# ------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "best_model.h5"))
CONFIDENCE_THRESHOLD = 0.40  # 40%

DEFAULT_CLASS_NAMES = [
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

def load_class_names():
    candidates = [
        os.environ.get("CLASS_INDICES_PATH"),
        BASE_DIR / "class_indices.json",
        BASE_DIR / "Main_folder" / "class_indices.json",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                class_indices = json.load(f)
            if isinstance(class_indices, dict) and class_indices:
                return [
                    name for name, _ in sorted(
                        class_indices.items(),
                        key=lambda item: int(item[1])
                    )
                ]
        except Exception as e:
            print(f"Could not load class indices from {path}: {e}")
    return DEFAULT_CLASS_NAMES

CLASS_NAMES = load_class_names()

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
        img = cv2.imread(str(image_path))
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

    image_path = upload_path(fname)
    if image_path is None or not image_path.exists():
        return "No image", 0.0, None

    # Step 1: Check if the image looks like a leaf
    is_leaf, reason = is_leaf_image(image_path)
    if not is_leaf:
        return reason, 0.0, None

    # Step 2: Run the model
    try:
        img = tf.keras.preprocessing.image.load_img(str(image_path), target_size=MODEL_IMG_SIZE)
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.expand_dims(input_arr, axis=0).astype("float32") / 255.0

        preds = MODEL.predict(input_arr)
        scores = preds[0]
        probs = _safe_softmax(scores)
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        confidence = float(probs[idx]) * 100.0
    except Exception as e:
        return f"Could not process image: {e}", 0.0, None

    if confidence < (CONFIDENCE_THRESHOLD * 100.0):
        return "Uncertain - Image may not be a crop leaf", confidence, None

    treatment = TREATMENT_DICT.get(label, "No treatment info available")
    return label, round(confidence, 2), treatment

def render_result(fname):
    image_path = upload_path(fname)
    if image_path is None or not image_path.exists():
        flash("No image to display.")
        return redirect('/input')
    label, confidence, treatment = processing(fname)
    return render_template('display.html',
                           variable_name=fname,
                           label=label,
                           confidence=confidence,
                           treatment=treatment)

# ---------- Upload / Display ----------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    GET  -> show the last image from this browser session
    POST -> handle file upload
    """

    if request.method == 'GET':
        fname = session.get('last_filename')
        if fname:
            return render_result(fname)
        flash("No uploaded image found. Please upload an image.")
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
        filename = unique_upload_name(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)
        session['last_filename'] = filename
        flash("Image successfully uploaded.")
        return redirect(url_for('display_image', filename=filename))
    else:
        flash("Allowed image types are - png, jpg, jpeg, gif.")
        return redirect('/input')

@app.route('/display')
def display_image():
    filename = request.args.get('filename') or session.get('last_filename')
    if filename:
        session['last_filename'] = filename
        return render_result(filename)
    flash("No image to display.")
    return redirect('/input')

if __name__ == "__main__":
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}
    app.run(debug=debug)
