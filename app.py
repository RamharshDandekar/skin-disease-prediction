# ==============================
# Imports
# ==============================
from flask import Flask, render_template, request, url_for
import os
import cv2
import threading
import numpy as np
import torch
import gc
import tempfile
from uuid import uuid4
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import supervision as sv
from supervision.annotators.core import BoxAnnotator


IMG_SIZE = int(os.environ.get("IMG_SIZE", "320"))
TILE_SIZE = (IMG_SIZE, IMG_SIZE)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
ENABLE_VIDEO = os.environ.get("ENABLE_VIDEO", "0") == "1"
VIDEO_FRAME_STRIDE = int(os.environ.get("VIDEO_FRAME_STRIDE", "4"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "180"))


# ==============================
# Flask App Initialization
# ==============================
app = Flask(__name__)

# Keep CPU usage predictable on small Render instances.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

# Lazy-loaded singleton model (loaded once per worker).
_MODEL = None
_MODEL_LOCK = threading.Lock()
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")
PREDICT_ARGS = {
    "device": "cpu",
    "imgsz": IMG_SIZE,
    "verbose": False,
    "fuse": False,  # avoid heavy conv+bn fusion on first run
}


# ==============================
# Configuration
# ==============================
# Allow overriding folders via environment (useful on Render / cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DEFAULT_OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", DEFAULT_UPLOAD_FOLDER)
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", DEFAULT_OUTPUT_FOLDER)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Ensure folders exist even when running under gunicorn (Render)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Render filesystem note: keep Ultralytics config in a writable temp location.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "Ultralytics"))


# ==============================
# Clinical knowledge base
# ==============================
DISEASE_INFO = {
    # Keys MUST match model.model.names exactly:
    # {0: 'AtopicDermatitis', 1: 'Leprosy', 2: 'Psoriasis', 3: 'acne', 4: 'keratosis pilaris', 5: 'wart'}
    "AtopicDermatitis": {
        "description": "A chronic, relapsing inflammatory eczema often associated with atopy (asthma, allergic rhinitis).",
        "symptoms": "Intensely itchy, red, scaly or lichenified plaques, commonly in flexural areas; may ooze or crust during flares.",
        "precautions": "Use regular emollients, avoid known irritants and harsh soaps, keep nails short, manage triggers such as heat and stress, and use prescribed topical anti‑inflammatories during flares."
    },
    "Leprosy": {
        "description": "A chronic infectious disease caused by Mycobacterium leprae, affecting skin, peripheral nerves, and sometimes eyes and mucosa.",
        "symptoms": "Hypopigmented or reddish skin patches with reduced sensation, numbness or tingling in hands/feet, muscle weakness, nerve thickening.",
        "precautions": "Seek specialist care promptly for multidrug therapy, avoid self‑medication with steroids, protect anesthetic areas from injury, and screen close contacts as advised by health authorities."
    },
    "Psoriasis": {
        "description": "A chronic immune‑mediated skin disease characterized by sharply demarcated, scaly plaques.",
        "symptoms": "Well‑defined red plaques with silvery scale on scalp, elbows, knees, or trunk; nail pitting; possible joint pain (psoriatic arthritis).",
        "precautions": "Keep skin moisturized, avoid skin trauma and harsh irritants, manage weight and stress, avoid smoking and excess alcohol, and follow dermatology guidance for topical, phototherapy, or systemic treatments."
    },
    "acne": {
        "description": "An inflammatory disorder of the pilosebaceous unit, common in adolescents and young adults.",
        "symptoms": "Whiteheads, blackheads, papules, pustules, nodules or cysts on face, chest, and back; potential scarring and post‑inflammatory marks.",
        "precautions": "Cleanse gently twice daily, avoid picking or squeezing lesions, use non‑comedogenic products, and seek medical care for persistent, nodulocystic, or scarring acne for appropriate topical or systemic therapy."
    },
    "keratosis pilaris": {
        "description": "A benign keratinization disorder causing rough, small papules around hair follicles.",
        "symptoms": "Dry, rough, 'chicken‑skin' bumps on outer arms, thighs, cheeks, or buttocks; often asymptomatic but may itch mildly.",
        "precautions": "Use regular moisturizers with gentle keratolytics (such as urea, lactic acid, or salicylic acid) as advised, avoid aggressive scrubbing, and maintain gentle skin care; usually improves with age."
    },
    "wart": {
        "description": "A benign viral proliferation of skin due to human papillomavirus (HPV) infection.",
        "symptoms": "Rough, hyperkeratotic papules or plaques, sometimes with black dots (thrombosed capillaries); plantar warts may be painful on pressure.",
        "precautions": "Avoid picking or biting, do not share personal items, keep affected areas dry, and use topical keratolytics or cryotherapy under medical supervision when treatment is needed."
    },
}


# ==============================
# Helper Functions
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_names(names_obj) -> dict[int, str]:
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names_obj)}
    return {}


def get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                resolved = MODEL_PATH
                if not os.path.exists(resolved):
                    if resolved.endswith(".onnx") and os.path.exists("best.pt"):
                        resolved = "best.pt"
                    else:
                        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
                _MODEL = YOLO(resolved)
    return _MODEL


def warmup_model() -> None:
    try:
        warmup = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        get_model().predict(source=warmup, **PREDICT_ARGS)
    except Exception as exc:
        print(f"Model warmup skipped: {exc}", flush=True)


if os.environ.get("WARMUP_MODEL", "1") == "1":
    warmup_model()


def summarize_detections(detections, names: dict[int, str]) -> list[dict]:
    """Return per‑class best confidence summary."""
    best_per_class = {}
    class_ids = detections.class_id if detections.class_id is not None else []
    confidences = detections.confidence if detections.confidence is not None else []

    for class_id, conf in zip(class_ids, confidences):
        idx = int(class_id)
        label = names.get(idx, f"class_{idx}")
        current = best_per_class.get(label)
        if current is None or conf > current["confidence"]:
            best_per_class[label] = {"label": label, "confidence": float(conf)}

    summary = sorted(
        best_per_class.values(),
        key=lambda x: x["confidence"],
        reverse=True,
    )
    return summary


def build_clinical_insights(disease_summary: list[dict]) -> list[dict]:
    """Attach clinical text (description, symptoms, precautions) to each predicted disease."""
    insights: list[dict] = []

    for entry in disease_summary:
        info = DISEASE_INFO.get(entry["label"])
        if not info:
            continue
        enriched = {
            "label": entry["label"],
            "confidence": entry["confidence"],
            "description": info["description"],
            "symptoms": info["symptoms"],
            "precautions": info["precautions"],
        }
        insights.append(enriched)

    return insights


# ==============================
# Image Processing
# ==============================
def process_image(input_image_path, output_image_path):
    model = get_model()

    image = cv2.imread(input_image_path)

    if image is None:
        print("Error reading image")
        return []

    resized = cv2.resize(image, TILE_SIZE)

    # YOLO detection
    results = model.predict(
        source=resized,
        **PREDICT_ARGS,
    )[0]
    names = normalize_names(results.names)

    detections = sv.Detections.from_ultralytics(results)

    box_annotator = BoxAnnotator()

    annotated = box_annotator.annotate(
        scene=resized,
        detections=detections,
    )

    # Draw label at center of box
    for xyxy, class_id, confidence in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence,
    ):

        x1, y1, x2, y2 = map(int, xyxy)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cls_idx = int(class_id)
        cls_name = names.get(cls_idx, f"class_{cls_idx}")
        label = f"{cls_name} {confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(output_image_path, annotated)

    print("Saved:", output_image_path)

    # Return summarized predictions for downstream insights
    summary = summarize_detections(detections, names)
    del image, resized, results, detections, annotated
    gc.collect()
    return summary


# ==============================
# Video Processing
# ==============================
def process_video(input_video_path, output_video_path):
    if not ENABLE_VIDEO:
        print("Video processing disabled for memory safety on free tier", flush=True)
        return

    model = get_model()

    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error opening video")
        return

    frame_width, frame_height = TILE_SIZE
    raw_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fps = max(1, raw_fps // max(1, VIDEO_FRAME_STRIDE))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    box_annotator = BoxAnnotator()

    frame_index = 0
    processed_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1
        if frame_index % max(1, VIDEO_FRAME_STRIDE) != 0:
            continue
        if processed_frames >= MAX_VIDEO_FRAMES:
            break

        resized = cv2.resize(frame, TILE_SIZE)

        results = model.predict(
            source=resized,
            **PREDICT_ARGS,
        )[0]
        names = normalize_names(results.names)

        detections = sv.Detections.from_ultralytics(results)

        annotated = box_annotator.annotate(
            scene=resized,
            detections=detections,
        )

        for xyxy, class_id, confidence in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
        ):

            x1, y1, x2, y2 = map(int, xyxy)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cls_idx = int(class_id)
            cls_name = names.get(cls_idx, f"class_{cls_idx}")
            label = f"{cls_name} {confidence:.2f}"

            cv2.putText(
                annotated,
                label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        out.write(annotated)
        processed_frames += 1

    cap.release()
    out.release()
    gc.collect()

    print("Saved video:", output_video_path)


# ==============================
# Flask Route
# ==============================
@app.route("/", methods=["GET", "POST"])
def upload_files():

    processed_items: list[dict] = []

    if request.method == "POST":

        files = request.files.getlist("files")

        for file in files:

            if file and allowed_file(file.filename):
                try:
                    safe_name = secure_filename(file.filename)
                    filepath = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        safe_name,
                    )

                    # Ensure the upload directory exists before saving
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    file.save(filepath)

                    extension = safe_name.rsplit(".", 1)[1].lower()

                    output_filename = f"annotated_{uuid4().hex}_{safe_name}"

                    output_path = os.path.join(
                        app.config["OUTPUT_FOLDER"],
                        output_filename,
                    )

                    item: dict = {
                        "url": url_for(
                            "static",
                            filename=f"outputs/{output_filename}",
                        ),
                        "is_video": extension in {"mp4", "avi", "mov"},
                        "diseases": [],
                        "insights": [],
                        "message": "",
                    }

                    if extension in {"png", "jpg", "jpeg"}:

                        # Get summarized predictions for this image
                        image_summary = process_image(filepath, output_path)
                        item["diseases"] = image_summary
                        item["insights"] = build_clinical_insights(image_summary)

                    elif extension in {"mp4", "avi", "mov"}:
                        if ENABLE_VIDEO:
                            process_video(filepath, output_path)
                        else:
                            item["message"] = "Video processing is disabled on this deployment to keep memory usage low."

                    processed_items.append(item)

                except Exception as exc:  # log and continue instead of crashing
                    print(f"Error processing file {file.filename}: {exc}", flush=True)

    return render_template(
        "index.html",
        processed_items=processed_items,
        enable_video=ENABLE_VIDEO,
    )


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=False)