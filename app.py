from flask import Flask, render_template, request, Response, flash, redirect, session, url_for, send_file, abort
import cv2
import json
import os
import sqlite3
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
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

DATABASE_PATH = Path(os.environ.get('DATABASE_PATH', INSTANCE_DIR / 'crop_history.db'))
app.secret_key = os.environ.get('SECRET_KEY', 'dev-cropdisease-change-me')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 64 * 1024 * 1024))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ENABLE_CAMERA = os.environ.get('ENABLE_CAMERA', '').lower() in {'1', 'true', 'yes', 'on'}

# Ensure shots folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

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

def crop_from_label(label):
    return label.split('___', 1)[0].replace('_', ' ').strip() if '___' in label else None

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                treatment TEXT
            )
        """)
        columns = {row['name'] for row in conn.execute("PRAGMA table_info(search_history)")}
        if 'crop' not in columns:
            conn.execute("ALTER TABLE search_history ADD COLUMN crop TEXT")
        for row in conn.execute("SELECT id, label FROM search_history WHERE crop IS NULL"):
            crop = crop_from_label(row['label'])
            if crop:
                conn.execute("UPDATE search_history SET crop = ? WHERE id = ?", (crop, row['id']))

def treatment_for(label):
    return TREATMENT_DICT.get(label, "No treatment info available")

def log_prediction(filename, label, confidence, treatment, crop=None):
    pending = session.pop('pending_prediction_log', None)
    if not pending or pending.get('filename') != filename:
        return
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO search_history (
                    created_at, filename, original_filename, source, label,
                    confidence, treatment, crop
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    filename,
                    pending.get('original_filename'),
                    pending.get('source', 'upload'),
                    label,
                    float(confidence or 0.0),
                    treatment,
                    crop or crop_from_label(label),
                )
            )
            conn.execute(
                """
                DELETE FROM search_history
                WHERE id NOT IN (
                    SELECT id FROM search_history ORDER BY id DESC LIMIT 1000
                )
                """
            )
    except sqlite3.Error as e:
        print(f"Search history log failed: {e}")

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
            log_prediction(fname, label, 0, json.dumps(assessment), assessment.get('crop'))
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

@app.route('/history')
def search_history():
    try:
        page = max(int(request.args.get('page', 1)), 1)
    except ValueError:
        page = 1
    per_page = 10
    crop = request.args.get('crop', '').strip()
    result = request.args.get('result', '').strip()
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    sort = request.args.get('sort', 'newest')
    sort_sql = {
        'newest': 'id DESC',
        'oldest': 'id ASC',
        'result': 'label COLLATE NOCASE ASC, id DESC',
        'crop': 'crop COLLATE NOCASE ASC, id DESC',
    }.get(sort, 'id DESC')
    filters = []
    params = []
    if crop:
        filters.append('crop = ?')
        params.append(crop)
    if result:
        filters.append('label LIKE ?')
        params.append(f'%{result}%')
    if date_from:
        filters.append('substr(created_at, 1, 10) >= ?')
        params.append(date_from)
    if date_to:
        filters.append('substr(created_at, 1, 10) <= ?')
        params.append(date_to)
    where = f"WHERE {' AND '.join(filters)}" if filters else ''
    with get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM search_history {where}", params).fetchone()[0]
        pages = max((total + per_page - 1) // per_page, 1)
        page = min(page, pages)
        searches = conn.execute(
            f"""SELECT id, created_at, filename, original_filename, source, label,
                       confidence, treatment, crop
                FROM search_history {where}
                ORDER BY {sort_sql} LIMIT ? OFFSET ?""",
            params + [per_page, (page - 1) * per_page]
        ).fetchall()
        crops = [row[0] for row in conn.execute(
            "SELECT DISTINCT crop FROM search_history WHERE crop IS NOT NULL ORDER BY crop COLLATE NOCASE"
        ).fetchall()]
    return render_template('history.html', searches=searches, crops=crops,
                           page=page, pages=pages, total=total,
                           crop=crop, result=result, date_from=date_from,
                           date_to=date_to, sort=sort)

def history_record(record_id):
    with get_db() as conn:
        return conn.execute(
            """SELECT id, created_at, filename, original_filename, source, label,
                      confidence, treatment, crop
               FROM search_history WHERE id = ?""", (record_id,)
        ).fetchone()

@app.route('/history/<int:record_id>')
def history_detail(record_id):
    search = history_record(record_id)
    if search is None:
        abort(404)
    return render_template('history_detail.html', search=search)

@app.route('/history/<int:record_id>/pdf')
def history_pdf(record_id):
    search = history_record(record_id)
    if search is None:
        abort(404)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError:
        abort(503, description='PDF export is not available on this installation.')
    image_path = upload_path(search['filename'])
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    pdf.setTitle(f"Crop analysis {search['id']}")
    pdf.setFont('Helvetica-Bold', 20)
    pdf.drawString(54, height - 60, 'Crop analysis report')
    pdf.setFont('Helvetica', 11)
    y = height - 90
    for title, value in (
        ('Date', search['created_at']), ('Crop', search['crop'] or 'Not recorded'),
        ('Result', search['label']), ('Confidence', f"{search['confidence']:.2f}%"),
    ):
        pdf.setFont('Helvetica-Bold', 11)
        pdf.drawString(54, y, f'{title}:')
        pdf.setFont('Helvetica', 11)
        pdf.drawString(135, y, str(value or ''))
        y -= 22
    if image_path and image_path.exists():
        try:
            pdf.drawImage(ImageReader(str(image_path)), 54, y - 220, width=300, height=200, preserveAspectRatio=True, anchor='c')
            y -= 245
        except (OSError, ValueError):
            pass
    pdf.setFont('Helvetica-Bold', 11)
    pdf.drawString(54, y, 'Treatment / notes')
    text = pdf.beginText(54, y - 18)
    text.setFont('Helvetica', 10)
    for line in str(search['treatment'] or 'No notes recorded.').splitlines():
        text.textLine(line[:110])
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True,
                     download_name=f'crop-analysis-{record_id}.pdf')

if __name__ == "__main__":
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}
    app.run(debug=debug)
