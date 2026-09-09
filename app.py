from flask import Flask, render_template, request, Response, flash, redirect, session, url_for
import cv2
import hmac
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from uuid import uuid4
from werkzeug.utils import secure_filename
import numpy as np
from prediction_policy import classify_scores, has_leaf_evidence
from leaf_locator import locate_leaf, crop_leaf
from PIL import Image
from crop_vision import CropVision, AnalysisError

ANALYSIS_PROVIDER = os.environ.get('ANALYSIS_PROVIDER', 'openai').lower()
if ANALYSIS_PROVIDER not in {'openai', 'local'}:
    raise RuntimeError('ANALYSIS_PROVIDER must be openai or local')
if ANALYSIS_PROVIDER == 'local':
    import tensorflow as tf
    from tensorflow.keras.models import load_model

# Globals
global switch
switch = 0

# Flask App Setup
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'shots'
INSTANCE_DIR = BASE_DIR / 'instance'
VISION = CropVision(INSTANCE_DIR)

@app.context_processor
def analysis_context():
    return {'analysis_provider': ANALYSIS_PROVIDER}

DATABASE_PATH = Path(os.environ.get('DATABASE_PATH', INSTANCE_DIR / 'crop_admin.db'))
RETRAIN_LOG_DIR = INSTANCE_DIR / 'retrain_logs'
TRAINING_DATA_ROOT = Path(os.environ.get('TRAINING_DATA_ROOT', INSTANCE_DIR / 'training_data'))
TRAINING_SCRIPT_PATH = Path(os.environ.get('TRAINING_SCRIPT_PATH', BASE_DIR / 'training' / 'train_model.py'))
app.secret_key = os.environ.get('SECRET_KEY', 'dev-cropdisease-change-me')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 64 * 1024 * 1024))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ENABLE_CAMERA = os.environ.get('ENABLE_CAMERA', '').lower() in {'1', 'true', 'yes', 'on'}
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Ensure shots folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
RETRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)

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
BUNDLED_MODEL_PATH = Path(os.environ.get("BUNDLED_MODEL_PATH", BASE_DIR / "best_model.h5"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", INSTANCE_DIR / "models" / "best_model.h5"))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RUNTIME_CLASS_INDICES_PATH = Path(os.environ.get("RUNTIME_CLASS_INDICES_PATH", INSTANCE_DIR / "class_indices.json"))
BUNDLED_CLASS_INDICES_PATH = Path(os.environ.get("BUNDLED_CLASS_INDICES_PATH", BASE_DIR / "class_indices.json"))
REQUIRE_NON_CROP_CLASS = os.environ.get("REQUIRE_NON_CROP_CLASS", "1").lower() in {"1", "true", "yes", "on"}
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
# Selected on public-data validation images; kept separate from non-leaf rejection.
DISEASE_CONFIDENCE_THRESHOLD = float(os.environ.get("DISEASE_CONFIDENCE_THRESHOLD", "0.95"))
PREDICTION_MARGIN_THRESHOLD = float(os.environ.get("PREDICTION_MARGIN_THRESHOLD", "0.20"))
NON_CROP_CLASS_NAME = "Not_A_Crop"
INVALID_IMAGE_LABEL = "image not supported"
UNCERTAIN_IMAGE_LABEL = "Uncertain image"

DEFAULT_CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    NON_CROP_CLASS_NAME,
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
        RUNTIME_CLASS_INDICES_PATH,
        BUNDLED_CLASS_INDICES_PATH,
        BASE_DIR / "Main_folder" / "class_indices.json",
    ]
    return load_class_names_from(candidates)

def load_class_names_from(candidates):
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
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use insecticides to control whiteflies, remove infected plants, and use resistant varieties.",
    "Tomato___healthy": "No disease-specific treatment was selected. Continue monitoring the plant.",
    "Tomato___Tomato_mosaic_virus": "Confirm the cause with a local crop specialist before choosing a treatment.",
    NON_CROP_CLASS_NAME: INVALID_IMAGE_LABEL
}

MODEL_LOCK = threading.Lock()

def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS treatments (
                label TEXT PRIMARY KEY,
                treatment TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                treatment TEXT,
                client_ip TEXT,
                user_agent TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                message TEXT,
                command TEXT,
                log_path TEXT
            )
        """)
        for label in CLASS_NAMES:
            conn.execute(
                """
                INSERT OR IGNORE INTO treatments (label, treatment, updated_at)
                VALUES (?, ?, ?)
                """,
                (label, TREATMENT_DICT.get(label, "No treatment info available"), utc_now())
            )

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash("Please log in as admin.")
            return redirect(url_for('admin_login', next=request.path))
        return view(*args, **kwargs)
    return wrapped

def treatment_for(label):
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT treatment FROM treatments WHERE label = ?",
                (label,)
            ).fetchone()
            if row:
                return row['treatment']
    except sqlite3.Error as e:
        print(f"Treatment lookup failed: {e}")
    return TREATMENT_DICT.get(label, "No treatment info available")

def log_prediction(filename, label, confidence, treatment):
    pending = session.pop('pending_prediction_log', None)
    if not pending or pending.get('filename') != filename:
        return
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    created_at, filename, original_filename, source, label,
                    confidence, treatment, client_ip, user_agent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    filename,
                    pending.get('original_filename'),
                    pending.get('source', 'upload'),
                    label,
                    float(confidence or 0.0),
                    treatment,
                    request.headers.get('X-Forwarded-For', request.remote_addr),
                    request.headers.get('User-Agent'),
                )
            )
    except sqlite3.Error as e:
        print(f"Prediction log failed: {e}")

def active_model_path():
    if _runtime_model_is_selectable():
        return MODEL_PATH
    return BUNDLED_MODEL_PATH

def reload_model():
    global MODEL, MODEL_IMG_SIZE, CLASS_NAMES
    model_path = active_model_path()
    loaded = load_model(str(model_path))
    class_names = _class_names_for_path(model_path)
    output_count = _model_output_count(loaded)
    if output_count is not None and output_count != len(class_names):
        if model_path == MODEL_PATH and BUNDLED_MODEL_PATH.exists():
            print(
                f"Ignoring runtime model because output count {output_count} "
                f"does not match {len(class_names)} class names."
            )
            model_path = BUNDLED_MODEL_PATH
            loaded = load_model(str(model_path))
            class_names = _class_names_for_path(model_path)
            output_count = _model_output_count(loaded)
        if output_count is not None and output_count != len(class_names):
            raise RuntimeError(
                f"Model output count {output_count} does not match {len(class_names)} class names."
            )
    with MODEL_LOCK:
        MODEL = loaded
        MODEL_IMG_SIZE = _infer_model_img_size(MODEL, fallback=(224, 224))
        CLASS_NAMES = class_names

def _class_names_for_path(path):
    if Path(path) == MODEL_PATH:
        return load_class_names_from([os.environ.get("CLASS_INDICES_PATH"), RUNTIME_CLASS_INDICES_PATH])
    return load_class_names_from([os.environ.get("CLASS_INDICES_PATH"), BUNDLED_CLASS_INDICES_PATH])

def _model_output_count(model):
    try:
        return int(model.output_shape[-1])
    except Exception:
        return None

def _runtime_model_is_selectable():
    if not MODEL_PATH.exists():
        return False
    if not RUNTIME_CLASS_INDICES_PATH.exists() and not os.environ.get("CLASS_INDICES_PATH"):
        print(f"Ignoring runtime model because {RUNTIME_CLASS_INDICES_PATH} is missing.")
        return False
    runtime_names = _class_names_for_path(MODEL_PATH)
    if REQUIRE_NON_CROP_CLASS and NON_CROP_CLASS_NAME not in runtime_names:
        print(f"Ignoring runtime model because {RUNTIME_CLASS_INDICES_PATH} does not include {NON_CROP_CLASS_NAME}.")
        return False
    return True

def mark_retrain_job(job_id, status, message, finished=False):
    with get_db() as conn:
        if finished:
            conn.execute(
                "UPDATE retrain_jobs SET status = ?, message = ?, finished_at = ? WHERE id = ?",
                (status, message, utc_now(), job_id)
            )
        else:
            conn.execute(
                "UPDATE retrain_jobs SET status = ?, message = ? WHERE id = ?",
                (status, message, job_id)
            )

def ensure_training_dirs():
    for split in ('train', 'val'):
        for label in CLASS_NAMES:
            (TRAINING_DATA_ROOT / 'dataset_split' / split / label).mkdir(parents=True, exist_ok=True)

def count_training_images(directory):
    if not directory.exists():
        return 0
    return sum(
        1 for path in directory.iterdir()
        if path.is_file() and allowed_file(path.name)
    )

def training_data_counts():
    ensure_training_dirs()
    rows = []
    for label in CLASS_NAMES:
        train_count = count_training_images(TRAINING_DATA_ROOT / 'dataset_split' / 'train' / label)
        val_count = count_training_images(TRAINING_DATA_ROOT / 'dataset_split' / 'val' / label)
        rows.append({'label': label, 'train': train_count, 'val': val_count})
    return rows

def dataset_counts_ready(counts):
    return bool(counts) and all(row['train'] > 0 and row['val'] > 0 for row in counts)

def local_main_folder_counts():
    train_root = BASE_DIR / "Main_folder" / "dataset_split" / "train"
    val_root = BASE_DIR / "Main_folder" / "dataset_split" / "val"
    if not train_root.exists() or not val_root.exists():
        return []
    rows = []
    for label in CLASS_NAMES:
        rows.append({
            'label': label,
            'train': count_training_images(train_root / label),
            'val': count_training_images(val_root / label),
        })
    return rows

def retrain_source():
    uploaded_counts = training_data_counts()
    if dataset_counts_ready(uploaded_counts):
        return {
            'train_dir': TRAINING_DATA_ROOT / 'dataset_split' / 'train',
            'val_dir': TRAINING_DATA_ROOT / 'dataset_split' / 'val',
            'source': 'admin uploaded training data',
            'counts': uploaded_counts,
        }

    local_counts = local_main_folder_counts()
    if dataset_counts_ready(local_counts):
        return {
            'train_dir': BASE_DIR / "Main_folder" / "dataset_split" / "train",
            'val_dir': BASE_DIR / "Main_folder" / "dataset_split" / "val",
            'source': 'local Main_folder training data',
            'counts': local_counts,
        }
    return None

def retrain_status():
    if ANALYSIS_PROVIDER == 'openai':
        return 'OpenAI active', 'Local retraining does not change OpenAI image analysis.', False, []
    counts = training_data_counts()
    source = retrain_source()
    if not TRAINING_SCRIPT_PATH.exists():
        return "Unavailable", "Training script is missing.", False, counts
    if source:
        return "Ready", f"Using {source['source']}.", True, counts
    return (
        "Needs Data",
        "Upload at least one training and validation image for every class.",
        False,
        counts,
    )

def save_training_files(label, split, files):
    if label not in CLASS_NAMES:
        return 0, "Unknown crop disease class."
    if split not in {'train', 'val'}:
        return 0, "Unknown dataset split."

    target_dir = TRAINING_DATA_ROOT / 'dataset_split' / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uploaded in files:
        if not uploaded or not uploaded.filename:
            continue
        if not allowed_file(uploaded.filename):
            continue
        filename = unique_upload_name(uploaded.filename)
        uploaded.save(target_dir / filename)
        saved += 1
    if saved == 0:
        return 0, "No valid image files were uploaded."
    return saved, f"Uploaded {saved} image(s) to {split} for {label}."

def run_retrain_job(job_id, command, log_path, env, pending_model_path):
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            process = subprocess.run(
                command,
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
        if process.returncode != 0:
            mark_retrain_job(job_id, "FAILED", f"Training exited with code {process.returncode}.", finished=True)
            return
        pending_model = Path(pending_model_path)
        if not pending_model.exists() or pending_model.stat().st_size < 1000000:
            mark_retrain_job(job_id, "FAILED", "Training did not produce a valid model file.", finished=True)
            return
        pending_model.replace(MODEL_PATH)
        reload_model()
        mark_retrain_job(job_id, "SUCCESS", "Training completed and the model was reloaded.", finished=True)
    except Exception as e:
        mark_retrain_job(job_id, "FAILED", str(e), finished=True)

def start_retrain_job():
    if ANALYSIS_PROVIDER == 'openai':
        return False, 'Local retraining is disabled while OpenAI analysis is active.'
    if not TRAINING_SCRIPT_PATH.exists():
        return False, "Training script is not available in this environment."
    source = retrain_source()
    if not source:
        return False, "Training data is incomplete. Upload train and validation images for every class first."

    with get_db() as conn:
        active = conn.execute(
            "SELECT id FROM retrain_jobs WHERE status = 'RUNNING' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if active:
            return False, f"Retraining job #{active['id']} is already running."
        pending_model_path = MODEL_PATH.with_name("best_model.pending.h5")
        initial_model_path = active_model_path()
        initial_class_indices_path = (
            RUNTIME_CLASS_INDICES_PATH if initial_model_path == MODEL_PATH else BUNDLED_CLASS_INDICES_PATH
        )
        env = os.environ.copy()
        env.update({
            "TRAIN_DIR": str(source['train_dir']),
            "VAL_DIR": str(source['val_dir']),
            "MODEL_OUTPUT": str(pending_model_path),
            "INITIAL_MODEL_PATH": str(initial_model_path),
            "INITIAL_CLASS_INDICES_PATH": str(initial_class_indices_path),
            "CLASS_INDICES_OUTPUT": str(INSTANCE_DIR / "class_indices.json"),
            "TRAINING_HISTORY_OUTPUT": str(RETRAIN_LOG_DIR / "training_history.pkl"),
            "RETRAIN_REPLACE_CLASSIFIER_HEAD": os.environ.get("RETRAIN_REPLACE_CLASSIFIER_HEAD", "1"),
            "RETRAIN_EPOCHS": os.environ.get("RETRAIN_EPOCHS", "10"),
            "RETRAIN_BATCH_SIZE": os.environ.get("RETRAIN_BATCH_SIZE", "16"),
        })
        command = [sys.executable, str(TRAINING_SCRIPT_PATH)]
        log_path = RETRAIN_LOG_DIR / f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        cursor = conn.execute(
            """
            INSERT INTO retrain_jobs (started_at, status, message, command, log_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now(), "RUNNING", "Training started.", " ".join(command), str(log_path))
        )
        job_id = cursor.lastrowid

    thread = threading.Thread(
        target=run_retrain_job,
        args=(job_id, command, str(log_path), env, str(pending_model_path)),
        daemon=True
    )
    thread.start()
    return True, f"Retraining job #{job_id} started."

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

MODEL = None
MODEL_IMG_SIZE = (224, 224)
if ANALYSIS_PROVIDER == 'local':
    try:
        startup_model_path = active_model_path()
        MODEL = load_model(str(startup_model_path))
        CLASS_NAMES = _class_names_for_path(startup_model_path)
        output_count = _model_output_count(MODEL)
        if output_count is not None and output_count != len(CLASS_NAMES):
            if startup_model_path == MODEL_PATH and BUNDLED_MODEL_PATH.exists():
                print(
                    f"Ignoring runtime model because output count {output_count} "
                    f"does not match {len(CLASS_NAMES)} class names."
                )
                startup_model_path = BUNDLED_MODEL_PATH
                MODEL = load_model(str(startup_model_path))
                CLASS_NAMES = _class_names_for_path(startup_model_path)
                output_count = _model_output_count(MODEL)
            if output_count is not None and output_count != len(CLASS_NAMES):
                raise RuntimeError(
                    f"Model output count {output_count} does not match {len(CLASS_NAMES)} class names."
                )
        MODEL_IMG_SIZE = _infer_model_img_size(MODEL, fallback=(224, 224))
        print(f"Loaded model: {startup_model_path}")
        print(f"Inferred model input size: {MODEL_IMG_SIZE}")
    except Exception as e:
        MODEL = None
        MODEL_IMG_SIZE = (224, 224)
        print(f"Error loading model: {e}")

# A separately evaluated locator can be enabled explicitly after its pipeline
# evaluation. Keeping it optional preserves the installed classifier by default.
LEAF_LOCATOR = None
LEAF_LOCATOR_THRESHOLD = float(os.environ.get('LEAF_LOCATOR_THRESHOLD', '0.90'))
if ANALYSIS_PROVIDER == 'local' and os.environ.get('LEAF_LOCATOR_PATH'):
    try:
        LEAF_LOCATOR = load_model(os.environ['LEAF_LOCATOR_PATH'], compile=False)
        print('Loaded optional dominant-leaf locator.')
    except Exception as error:
        raise RuntimeError('Configured leaf locator could not be loaded') from error

init_db()

def processing(fname):
    global MODEL
    if MODEL is None:
        return "Model not loaded", 0.0, None

    image_path = upload_path(fname)
    if image_path is None or not image_path.exists():
        return "No image", 0.0, None

    # Let the trained classifier evaluate the image; color is not a leaf detector.
    try:
        if LEAF_LOCATOR is not None:
            with Image.open(image_path) as original:
                rgb = original.convert('RGB')
                with MODEL_LOCK:
                    box = locate_leaf(LEAF_LOCATOR, rgb, LEAF_LOCATOR_THRESHOLD)
                if box is None:
                    return (UNCERTAIN_IMAGE_LABEL, 0.0,
                            'A leaf could not be located reliably. Please photograph one leaf up close. '
                            'No disease-specific treatment has been selected.')
                img = crop_leaf(rgb, box).resize((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0]), Image.Resampling.NEAREST)
        else:
            img = tf.keras.preprocessing.image.load_img(str(image_path), target_size=MODEL_IMG_SIZE)
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.expand_dims(input_arr, axis=0).astype("float32") / 255.0

        with MODEL_LOCK:
            preds = MODEL.predict(input_arr)
            label, confidence = classify_scores(
                preds[0], CLASS_NAMES,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                margin_threshold=PREDICTION_MARGIN_THRESHOLD,
                non_crop_class=NON_CROP_CLASS_NAME,
                disease_confidence_threshold=DISEASE_CONFIDENCE_THRESHOLD,
            )
            leaf_evidence = has_leaf_evidence(preds[0], CLASS_NAMES)
    except Exception as e:
        return f"Could not process image: {e}", 0.0, None

    if label is None:
        if leaf_evidence:
            return (
                UNCERTAIN_IMAGE_LABEL,
                round(confidence, 2),
                "A leaf was recognized, but its condition could not be identified reliably. "
                "The condition may be outside the trained categories. No disease-specific "
                "treatment has been selected."
            )
        return (
            UNCERTAIN_IMAGE_LABEL,
            round(confidence, 2),
            "The model could not reliably distinguish a supported crop leaf from other images. "
            "Please retake the photo with one leaf in focus and a simple background."
        )

    if label == 'Unsupported_Leaf':
        return (
            UNCERTAIN_IMAGE_LABEL,
            round(confidence, 2),
            "A leaf was recognized, but the model could not identify a supported crop condition. "
            "No disease-specific treatment has been selected."
        )

    if label == NON_CROP_CLASS_NAME:
        return (
            INVALID_IMAGE_LABEL,
            round(confidence, 2),
            "The model could not recognize a supported crop condition in this image. "
            "This does not establish that the photo contains no leaf: unfamiliar leaf "
            "conditions can also be rejected. Try a close-up of one leaf; if it is still "
            "rejected, its condition may require additional training data."
        )

    treatment = treatment_for(label)
    return label, round(confidence, 2), treatment

def render_result(fname):
    image_path = upload_path(fname)
    if image_path is None or not image_path.exists():
        flash("No image to display.")
        return redirect('/input')
    if ANALYSIS_PROVIDER == 'openai':
        try:
            assessment = VISION.analyze(image_path)
            titles = {'healthy': 'No obvious symptoms', 'uncertain': 'More information needed',
                      'non_leaf': 'No leaf identified', 'unsupported_crop': 'Unsupported crop'}
            label = ('Possible ' + assessment['condition']) if assessment['status'] == 'possible_condition' else titles[assessment['status']]
            log_prediction(fname, label, 0, json.dumps(assessment))
            return render_template('display.html', variable_name=fname, label=label,
                                   confidence=None, treatment='', assessment=assessment)
        except AnalysisError as error:
            return render_template('display.html', variable_name=fname,
                                   label='Analysis unavailable', confidence=None,
                                   treatment=str(error), assessment=None)
    label, confidence, treatment = processing(fname)
    log_prediction(fname, label, confidence, treatment)
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
        session['pending_prediction_log'] = {
            'filename': filename,
            'original_filename': secure_filename(file.filename),
            'source': request.form.get('source', 'upload')
        }
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

# ---------- Admin ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if hmac.compare_digest(username, ADMIN_USERNAME) and hmac.compare_digest(password, ADMIN_PASSWORD):
            session['admin_logged_in'] = True
            flash("Admin login successful.")
            next_url = request.args.get('next') or url_for('admin_dashboard')
            if not next_url.startswith('/'):
                next_url = url_for('admin_dashboard')
            return redirect(next_url)
        flash("Invalid admin username or password.")
    return render_template('admin_login.html')

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.pop('admin_logged_in', None)
    flash("Admin logged out.")
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    with get_db() as conn:
        treatment_count = conn.execute("SELECT COUNT(*) AS total FROM treatments").fetchone()['total']
        prediction_count = conn.execute("SELECT COUNT(*) AS total FROM prediction_logs").fetchone()['total']
        recent_logs = conn.execute(
            """
            SELECT id, created_at, source, label, confidence
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT 8
            """
        ).fetchall()
        retrain_jobs = conn.execute(
            """
            SELECT id, started_at, finished_at, status, message
            FROM retrain_jobs
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()
    retrain_status_label, retrain_status_message, retrain_available, training_counts = retrain_status()
    total_train = sum(row['train'] for row in training_counts)
    total_val = sum(row['val'] for row in training_counts)
    return render_template(
        'admin_dashboard.html',
        treatment_count=treatment_count,
        prediction_count=prediction_count,
        recent_logs=recent_logs,
        retrain_jobs=retrain_jobs,
        retrain_available=retrain_available,
        retrain_status_label=retrain_status_label,
        retrain_status_message=retrain_status_message,
        training_counts=training_counts,
        total_train=total_train,
        total_val=total_val
    )

@app.route('/admin/training-data', methods=['GET', 'POST'])
@admin_required
def admin_training_data():
    if request.method == 'POST':
        label = request.form.get('label', '')
        split = request.form.get('split', '')
        files = request.files.getlist('files')
        _saved, message = save_training_files(label, split, files)
        flash(message)
        return redirect(url_for('admin_training_data'))

    retrain_status_label, retrain_status_message, retrain_available, counts = retrain_status()
    total_train = sum(row['train'] for row in counts)
    total_val = sum(row['val'] for row in counts)
    return render_template(
        'admin_training_data.html',
        class_names=CLASS_NAMES,
        counts=counts,
        retrain_status_label=retrain_status_label,
        retrain_status_message=retrain_status_message,
        retrain_available=retrain_available,
        total_train=total_train,
        total_val=total_val
    )

@app.route('/admin/treatments', methods=['GET', 'POST'])
@admin_required
def admin_treatments():
    if request.method == 'POST':
        label = request.form.get('label', '')
        treatment = request.form.get('treatment', '').strip()
        if label not in CLASS_NAMES:
            flash("Unknown crop disease class.")
        elif not treatment:
            flash("Treatment cannot be empty.")
        else:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO treatments (label, treatment, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(label) DO UPDATE SET
                        treatment = excluded.treatment,
                        updated_at = excluded.updated_at
                    """,
                    (label, treatment, utc_now())
                )
            flash("Treatment updated.")
        return redirect(url_for('admin_treatments'))

    with get_db() as conn:
        treatments = conn.execute(
            """
            SELECT label, treatment, updated_at
            FROM treatments
            ORDER BY label
            """
        ).fetchall()
    return render_template('admin_treatments.html', treatments=treatments)

@app.route('/admin/logs')
@admin_required
def admin_logs():
    try:
        limit = min(max(int(request.args.get('limit', 100)), 1), 500)
    except ValueError:
        limit = 100
    with get_db() as conn:
        logs = conn.execute(
            """
            SELECT id, created_at, filename, original_filename, source, label,
                   confidence, treatment, client_ip, user_agent
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    return render_template('admin_logs.html', logs=logs, limit=limit)

@app.route('/admin/retrain', methods=['POST'])
@admin_required
def admin_retrain():
    started, message = start_retrain_job()
    flash(message)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/retrain/log/<int:job_id>')
@admin_required
def admin_retrain_log(job_id):
    with get_db() as conn:
        job = conn.execute(
            "SELECT log_path FROM retrain_jobs WHERE id = ?",
            (job_id,)
        ).fetchone()
    if not job or not job['log_path']:
        return Response("Retrain log not found.", status=404, mimetype='text/plain')
    log_path = Path(job['log_path'])
    try:
        log_path.resolve().relative_to(RETRAIN_LOG_DIR.resolve())
    except ValueError:
        return Response("Invalid retrain log path.", status=400, mimetype='text/plain')
    if not log_path.exists():
        return Response("Retrain log is not available yet.", status=404, mimetype='text/plain')
    return Response(log_path.read_text(encoding='utf-8', errors='replace'), mimetype='text/plain')

if __name__ == "__main__":
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}
    app.run(debug=debug)
