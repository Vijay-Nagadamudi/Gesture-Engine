"""
Gesture Engine — Low-Latency Streamlit UI
==========================================

This UI is deliberately built around WebRTC instead of an OpenCV camera
loop inside Streamlit.

Performance choices:
- Browser camera -> WebRTC -> background video processor.
- No while-True loop.
- No st_autorefresh.
- No Streamlit rerun for every frame.
- Detector / feature extractor / predictor / renderer are created once
  per WebRTC stream, not once per frame.
- Camera input is capped at 640x360 / ~24 FPS to reduce CPU + network load.
- SVM prediction runs every 3rd frame; the latest prediction is reused
  between inference frames.
- The existing project's CV/ML classes are reused unchanged.
"""

import threading

import av
import cv2
import streamlit as st

try:
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
except ImportError:
    st.error(
        "streamlit-webrtc is not installed.\n\n"
        "Install it with:\n"
        "pip install streamlit-webrtc av"
    )
    st.stop()

from detection.hand_detector import HandDetector
from features.extractor import LandMarkFeatureExtractor
from models.predictor import GesturePredictor
from visualization.renderer import HandRenderer


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Gesture Engine",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

GESTURES = [
    "PINCH",
    "ITALIAN",
    "ILY",
    "ROCK",
    "CALL_ME",
    "FINGER_HEART",
    "OPEN_PALM",
    "FIST",
    "ONE",
    "TWO",
]

GESTURE_ICONS = {
    "PINCH": "🤏",
    "ITALIAN": "🤌",
    "ILY": "🤟",
    "ROCK": "🤘",
    "CALL_ME": "🤙",
    "FINGER_HEART": "🫰",
    "OPEN_PALM": "🖐️",
    "FIST": "✊",
    "ONE": "☝️",
    "TWO": "✌️",
}

# Main latency control.
INFER_EVERY_N_FRAMES = 3

# Keep browser camera modest. 640x360 is much cheaper than 1280x720
# and is more than enough for 21-point hand landmarks.
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
VIDEO_FPS = 24

COLOR_BLUE = (255, 140, 0)       # BGR
COLOR_GREEN = (80, 200, 80)
COLOR_AMBER = (0, 165, 255)
COLOR_GRAY = (145, 145, 145)
COLOR_DARK = (20, 20, 26)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, #151a2a 0%, #0c101a 48%, #080a10 100%);
    color: #E7EAF1;
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.ge-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 16px;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.ge-icon {
    font-size: 36px;
}

.ge-title {
    margin: 0;
    font-size: 30px;
    font-weight: 750;
    color: #F5F7FB;
}

.ge-subtitle {
    margin: 3px 0 0;
    color: #8D96AA;
    font-size: 14px;
}

.ge-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
}

.ge-card h4 {
    margin: 0 0 13px;
    color: #F4F6FB;
    font-size: 16px;
}

.ge-video {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.10);
    background: #080a10;
}

.ge-video video {
    border-radius: 15px !important;
}

.ge-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .3px;
}

.ge-online {
    color: #7EE787;
    background: rgba(46,160,67,.12);
    border: 1px solid rgba(63,185,80,.28);
}

.ge-offline {
    color: #9AA3B5;
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.08);
}

.ge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
}

.ge-kv {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255,255,255,.055);
    font-size: 13px;
}

.ge-kv:last-child {
    border-bottom: 0;
}

.ge-kv span:first-child {
    color: #8D96AA;
}

.ge-kv span:last-child {
    color: #E8EBF2;
    font-weight: 600;
}

.ge-gesture-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.ge-gesture {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 9px;
    border-radius: 9px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.06);
    color: #C9CFDC;
    font-size: 12px;
}

.ge-gesture .emoji {
    font-size: 17px;
}

.ge-pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    flex-wrap: wrap;
}

.ge-node {
    padding: 8px 11px;
    border-radius: 8px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.065);
    color: #C9CFDC;
    font-size: 12px;
}

.ge-arrow {
    color: #4C9EFF;
    font-weight: 700;
}

.ge-note {
    color: #747E92;
    font-size: 11px;
    line-height: 1.5;
}

div.stButton > button {
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,.10);
    background: rgba(255,255,255,.045);
    color: #E9ECF3;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: rgba(76,158,255,.55);
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class GestureVideoProcessor(VideoProcessorBase):
    """
    The complete real-time CV pipeline lives here.

    IMPORTANT:
    This class never calls Streamlit UI APIs. It runs independently on
    the WebRTC worker thread, which is the main reason this version is
    smoother than a Streamlit polling/camera loop.
    """

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = LandMarkFeatureExtractor()
        self.predictor = GesturePredictor()
        self.renderer = HandRenderer()

        self.frame_count = 0

        self.latest_gesture = None
        self.latest_confidence = 0.0
        self.hand_present = False

        self._lock = threading.Lock()

    @staticmethod
    def _get_first_hand(results):
        """
        Supports both shapes used by the project's earlier detector versions:
          1) results.hand_landmarks
          2) a direct list of landmarks
        """
        if results is None:
            return None

        if hasattr(results, "hand_landmarks"):
            hands = results.hand_landmarks
            if hands:
                return hands[0]
            return None

        if isinstance(results, (list, tuple)):
            return results[0] if results else None

        return None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # WebRTC may provide a larger frame than requested. Keep processing
        # resolution bounded so MediaPipe does not receive unnecessary pixels.
        h, w = img.shape[:2]
        if w > VIDEO_WIDTH:
            scale = VIDEO_WIDTH / float(w)
            img = cv2.resize(
                img,
                (VIDEO_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Existing project detector — no duplicated ML logic.
        results = self.detector.detect(img)
        landmarks = self._get_first_hand(results)

        hand_present = landmarks is not None

        if hand_present:
            # Draw skeleton every frame for responsive visual tracking.
            try:
                img = self.renderer.draw_landmarks(img, landmarks)
            except Exception:
                # If an older renderer expects a slightly different result
                # shape, keep the camera alive instead of killing the stream.
                pass

            # Prediction is intentionally throttled.
            if self.frame_count % INFER_EVERY_N_FRAMES == 0:
                try:
                    features = self.extractor.extract(landmarks)
                    gesture, confidence = self.predictor.predict(features)

                    with self._lock:
                        self.latest_gesture = gesture
                        self.latest_confidence = float(confidence or 0.0)
                except Exception:
                    # Keep the last stable prediction if one frame fails.
                    pass

        else:
            with self._lock:
                self.latest_gesture = None
                self.latest_confidence = 0.0

        with self._lock:
            self.hand_present = hand_present
            gesture = self.latest_gesture
            confidence = self.latest_confidence

        self._draw_overlay(img, gesture, confidence, hand_present)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @staticmethod
    def _draw_overlay(img, gesture, confidence, hand_present):
        h, w = img.shape[:2]

        if not hand_present:
            label = "NO HAND"
            color = COLOR_GRAY
        elif not gesture:
            label = "DETECTING..."
            color = COLOR_AMBER
        else:
            label = f"{gesture}   {confidence * 100:.0f}%"
            color = COLOR_BLUE

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2

        (tw, th), _ = cv2.getTextSize(
            label, font, scale, thickness
        )

        x0, y0 = 14, 14
        pad_x, pad_y = 10, 8
        x1 = x0 + tw + pad_x * 2
        y1 = y0 + th + pad_y * 2

        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            COLOR_DARK,
            -1,
        )
        cv2.addWeighted(
            overlay,
            0.72,
            img,
            0.28,
            0,
            img,
        )

        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            color,
            1,
        )

        cv2.putText(
            img,
            label,
            (x0 + pad_x, y1 - pad_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # LIVE indicator
        cv2.circle(
            img,
            (w - 22, 22),
            5,
            COLOR_GREEN,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "LIVE",
            (w - 65, 27),
            font,
            0.48,
            COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )

    def get_state(self):
        with self._lock:
            return (
                self.latest_gesture,
                self.latest_confidence,
                self.hand_present,
            )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="ge-header">
    <div class="ge-icon">🖐️</div>
    <div>
        <p class="ge-title">Gesture Engine</p>
        <p class="ge-subtitle">
            Real-time hand gesture recognition · low-latency WebRTC pipeline
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT
# ============================================================

video_col, info_col = st.columns([2.05, 1], gap="large")


# ============================================================
# LIVE CAMERA
# ============================================================

with video_col:
    st.markdown("### Live Recognition")

    st.markdown('<div class="ge-video">', unsafe_allow_html=True)

    webrtc_ctx = webrtc_streamer(
        key="gesture-engine-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=GestureVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": VIDEO_WIDTH, "max": VIDEO_WIDTH},
                "height": {"ideal": VIDEO_HEIGHT, "max": VIDEO_HEIGHT},
                "frameRate": {"ideal": VIDEO_FPS, "max": VIDEO_FPS},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if webrtc_ctx.state.playing:
        st.markdown(
            """
            <span class="ge-status ge-online">
                <span class="ge-dot"></span>
                STREAM ONLINE
            </span>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <span class="ge-status ge-offline">
                <span class="ge-dot"></span>
                STREAM OFFLINE · PRESS START
            </span>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "The video runs on the WebRTC worker thread. "
        "Streamlit is not rerun for every camera frame."
    )


# ============================================================
# INFORMATION PANEL
# ============================================================

with info_col:
    st.markdown(
        """
        <div class="ge-card">
            <h4>AI Model</h4>
            <div class="ge-kv">
                <span>Classifier</span>
                <span>SVM</span>
            </div>
            <div class="ge-kv">
                <span>Features</span>
                <span>63</span>
            </div>
            <div class="ge-kv">
                <span>Landmarks</span>
                <span>21</span>
            </div>
            <div class="ge-kv">
                <span>Gestures</span>
                <span>10</span>
            </div>
            <div class="ge-kv">
                <span>Inference</span>
                <span>Every 3rd frame</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gesture_html = "".join(
        f"""
        <div class="ge-gesture">
            <span class="emoji">{GESTURE_ICONS[g]}</span>
            <span>{g.replace("_", " ").title()}</span>
        </div>
        """
        for g in GESTURES
    )

    st.markdown(
        f"""
        <div class="ge-card">
            <h4>Supported Gestures</h4>
            <div class="ge-gesture-grid">
                {gesture_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PIPELINE
# ============================================================

st.markdown("### Processing Pipeline")

st.markdown(
    """
<div class="ge-card">
    <div class="ge-pipeline">
        <div class="ge-node">📷 Camera</div>
        <div class="ge-arrow">→</div>
        <div class="ge-node">✋ MediaPipe</div>
        <div class="ge-arrow">→</div>
        <div class="ge-node">📍 21 Landmarks</div>
        <div class="ge-arrow">→</div>
        <div class="ge-node">⚙️ 63 Features</div>
        <div class="ge-arrow">→</div>
        <div class="ge-node">🧠 SVM</div>
        <div class="ge-arrow">→</div>
        <div class="ge-node">🎯 Prediction</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL / ARCHITECTURE
# ============================================================

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        """
        <div class="ge-card">
            <h4>Architecture</h4>
            <div class="ge-kv"><span>Capture</span><span>Browser WebRTC</span></div>
            <div class="ge-kv"><span>Detection</span><span>MediaPipe</span></div>
            <div class="ge-kv"><span>Feature Extraction</span><span>Existing project extractor</span></div>
            <div class="ge-kv"><span>Classification</span><span>SVM</span></div>
            <div class="ge-kv"><span>Rendering</span><span>OpenCV overlay</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="ge-card">
            <h4>Performance Design</h4>
            <div class="ge-kv"><span>Camera</span><span>640 × 360</span></div>
            <div class="ge-kv"><span>Target FPS</span><span>24</span></div>
            <div class="ge-kv"><span>SVM inference</span><span>Every 3 frames</span></div>
            <div class="ge-kv"><span>Stream reruns</span><span>None per frame</span></div>
            <div class="ge-kv"><span>Processing thread</span><span>WebRTC worker</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="ge-note">
    Gesture Engine · The UI layer reuses the existing detection, feature
    extraction, prediction and rendering components. The camera transport
    and frame scheduling are handled by WebRTC to avoid the lag caused by
    Streamlit polling loops and full-script reruns.
</div>
""",
    unsafe_allow_html=True,
)
