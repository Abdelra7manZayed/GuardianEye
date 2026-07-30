import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import shutil
from ultralytics import YOLO

st.set_page_config(page_title="PPE Detection", layout="wide")
st.title("🦺 PPE Violation Detection")

unsafe_classes = ["no_helmet", "no_vest", "no_vest_no_helmet"]


def find_best_model(search_dir="runs"):
    """Search for the most recently modified best.pt under search_dir."""
    candidates = []
    if os.path.exists(search_dir):
        for root, dirs, files in os.walk(search_dir):
            if "best.pt" in files:
                full_path = os.path.join(root, "best.pt")
                candidates.append((os.path.getmtime(full_path), full_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # most recently modified first
    return candidates[0][1]


@st.cache_resource
def load_model(path):
    return YOLO(path)


def get_working_fourcc(output_path, fps, width, height):
    """Try browser-friendly codecs first, fall back to mp4v if unavailable."""
    for codec in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def process_video(input_path, model, conf_threshold,
                   blur_strength, test_interval_seconds, test_burst_frames,
                   progress_bar, status_text, debug_text):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = get_working_fourcc(output_path, fps, width, height)
    if out is None:
        cap.release()
        return None

    # --- Frame sampling: every `test_interval_seconds`, run detection on a
    # burst of `test_burst_frames` frames, then skip the rest of the cycle
    # untouched (no detection, no blur/box, frame written as-is). ---
    interval_frames = max(1, round(test_interval_seconds * fps))
    burst_frames = max(1, min(int(test_burst_frames), interval_frames))
    # Each tested frame stands in for this many real frames time-wise, so the
    # violation score (and its /fps -> seconds conversion) stays a fair
    # estimate of real elapsed time even though we only sample a slice.
    weight_per_test = interval_frames / burst_frames

    frame_idx = 0

    # Per-object violation score, measured in FRAME-EQUIVALENTS: {track_id: score}
    violation_scores = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        position_in_cycle = (frame_idx - 1) % interval_frames
        should_test = position_in_cycle < burst_frames

        if not should_test:
            # Skip detection entirely this frame — just pass it through.
            out.write(frame)
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.text(f"Frame {frame_idx}/{total_frames} — skipped (sampling)")
            continue

        # Detection runs only on the sampled "burst" frames
        results = model.track(
            frame, conf=conf_threshold, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )

        display_frame = frame.copy()
        seen_ids_this_frame = set()
        unsafe_detections = []  # (box, track_id, cls_name, conf, new_score)

        for r in results:
            if r.boxes.id is None:
                continue

            for box, track_id in zip(r.boxes, r.boxes.id):
                track_id = int(track_id)
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                is_unsafe = cls_name in unsafe_classes

                seen_ids_this_frame.add(track_id)
                prev_score = violation_scores.get(track_id, 0)

                if is_unsafe:
                    new_score = prev_score + weight_per_test
                else:
                    new_score = max(0, prev_score - weight_per_test)
                violation_scores[track_id] = new_score

                if is_unsafe:
                    conf = float(box.conf[0])
                    unsafe_detections.append((box, track_id, cls_name, conf, new_score))



        # Blur the whole frame once if there's at least one violator,
        # then restore + box each violator's own region so they stay sharp
        if unsafe_detections:
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            display_frame = cv2.GaussianBlur(display_frame, (k, k), 0)

            for box, track_id, cls_name, conf, new_score in unsafe_detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                seconds_so_far = new_score / fps
                label = f"ID {track_id} | {cls_name} {conf:.2f} | {seconds_so_far:.1f}s"
                cv2.putText(
                    display_frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

        out.write(display_frame)

        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
        status_text.text(f"Frame {frame_idx}/{total_frames} — testing")

    cap.release()
    out.release()
    return output_path


# ---------- Settings (sidebar) ----------
st.sidebar.header("Settings")

auto_path = find_best_model()
if auto_path:
    st.sidebar.success(f"Model found automatically:\n{auto_path}")
else:
    st.sidebar.warning("No best.pt found automatically under 'runs/'.")

model_path = st.sidebar.text_input(
    "Model path (best.pt) — auto-filled, edit only if wrong",
    value=auto_path or "runs/train/ppe_yolo11s/weights/best.pt"
)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
blur_strength = st.sidebar.select_slider(
    "Blur strength", options=[5, 9, 15, 21, 25, 35], value=15
)

st.sidebar.subheader("Frame Sampling (reduces load on the model)")
test_interval_seconds = st.sidebar.slider(
    "Test interval (seconds)", 1.0, 10.0, 3.0, 0.5
)
test_burst_frames = st.sidebar.slider(
    "Frames tested per burst", 1, 15, 5, 1
)

# ---------- Session state init ----------
if "output_path" not in st.session_state:
    st.session_state.output_path = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

# If a new file is uploaded, clear any previous result so it doesn't show stale output
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.output_path = None
    st.session_state.last_uploaded_name = uploaded_file.name

process_clicked = st.button("▶ Process Video", disabled=uploaded_file is None)

if process_clicked:
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    model = load_model(model_path)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    progress_bar = st.progress(0)
    status_text = st.empty()
    debug_text = st.empty()

    output_path = process_video(
        input_path, model, conf_threshold,
        blur_strength, test_interval_seconds, test_burst_frames,
        progress_bar, status_text, debug_text
    )

    try:
        os.remove(input_path)
    except PermissionError:
        pass

    if output_path is None:
        st.error("Could not create output video — no working video codec found on this system.")
    else:
        status_text.text("Done.")
        st.session_state.output_path = output_path

# ---------- Show result (persists across reruns, e.g. clicking Download) ----------
if st.session_state.output_path and os.path.exists(st.session_state.output_path):
    st.subheader("Output video")
    st.video(st.session_state.output_path)

    with open(st.session_state.output_path, "rb") as f:
        st.download_button(
            "Download processed video",
            data=f,
            file_name="ppe_detection_output.mp4",
            mime="video/mp4"
        )
elif uploaded_file is None:
    st.info("Upload a video, then click 'Process Video' to start detection.")

# ---------- Team ----------
st.divider()
st.markdown("### 👥 Team")

team = [
    ("Mohamed Mahmoud Salem", "https://www.linkedin.com/in/mohamed-mahmoud-mohamed-salem?utm_source=share_via&utm_content=profile&utm_medium=member_android"),
    ("Ahmed Ayman", "https://www.linkedin.com/in/ahmed-ayman-b1648232b?utm_source=share_via&utm_content=profile&utm_medium=member_android"),
    ("Antwan Gamil", "https://www.linkedin.com/in/antwan-gamil?utm_source=share_via&utm_content=profile&utm_medium=member_android"),
    ("Abdelrahman Gamal Zayed", "https://www.linkedin.com/in/abdelrahman-gamal-zayed?utm_source=share_via&utm_content=profile&utm_medium=member_android"),
    ("Omar Saber", "https://www.linkedin.com/in/omar-saber1?utm_source=share_via&utm_content=profile&utm_medium=member_android"),
]

team_cols = st.columns(len(team))
for col, (name, url) in zip(team_cols, team):
    with col:
        st.link_button(f"🔗 {name}", url, use_container_width=True)