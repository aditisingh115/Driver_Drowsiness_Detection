import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import threading
import winsound
import av
import csv
import io
import joblib
MODEL_PATH = "models/eye_state_model.pkl"
@st.cache_resource
def load_eye_model():
    return joblib.load(MODEL_PATH)

eye_model = load_eye_model()

from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="DrowsyGuard",
    page_icon="🚗",
    layout="wide"
)

EAR_THRESHOLD = 0.21
CLOSED_EYE_SECONDS = 1.0

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# -------------------- DETECTION DATA --------------------

class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.alert_count = 0
        self.history = []
        self.last_status = "Waiting"
        self.last_ear = None

    def update_status(self, status, ear=None):
        with self.lock:
            self.last_status = status
            self.last_ear = ear

    def add_alert(self):
        with self.lock:
            self.alert_count += 1
            self.history.append({
                "Alert No.": self.alert_count,
                "Date & Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Event": "Drowsiness Detected"
            })

    def get_data(self):
        with self.lock:
            return (
                self.alert_count,
                list(self.history),
                self.last_status,
                self.last_ear
            )


if "detection_state" not in st.session_state:
    st.session_state.detection_state = DetectionState()

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False


# -------------------- THEME --------------------

st.sidebar.markdown(
    "<h1 style='text-align:center;'>🚗 DrowsyGuard</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.selectbox(
    "🎨 Choose Theme",
    ["Dark", "Light"],
    key="theme"
)

if st.session_state.theme == "Dark":
    background = "#0f172a"
    card = "#1e293b"
    text_color = "#ffffff"
    accent = "#a78bfa"
else:
    background = "#f1f5f9"
    card = "#ffffff"
    text_color = "#172033"
    accent = "#6d28d9"

st.markdown(f"""
<style>
.stApp {{
    background: {background};
    color: {text_color};
}}
[data-testid="stSidebar"] {{
    background: {card};
}}
[data-testid="stSidebar"] * {{
    color: {text_color};
}}
h1, h2, h3, p, label {{
    color: {text_color};
}}
div[data-testid="stMetric"] {{
    background: {card};
    padding: 18px;
    border-radius: 12px;
    border: 1px solid {accent};
}}
.block-container {{
    max-width: 1150px;
    padding-top: 2rem;
}}
</style>
""", unsafe_allow_html=True)


# -------------------- NAVIGATION --------------------

page = st.sidebar.radio(
    "✨ Navigation Menu",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📜 Detection History",
        "🤖 About Project",
        "🛡️ Safety Tips"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-Powered Driver Safety System")
st.sidebar.caption("Python | OpenCV | MediaPipe")


# -------------------- EAR CALCULATION --------------------

def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = [
        np.array(point) for point in points
    ]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# -------------------- DROWSINESS DETECTOR --------------------

class DrowsinessDetector(VideoProcessorBase):

    def __init__(self, shared_state):
        self.shared_state = shared_state

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.closed_since = None
        self.alarm_active = False
        self.alert_recorded = False
        self.lock = threading.Lock()

    def play_alarm(self):
        while True:
            with self.lock:
                if not self.alarm_active:
                    break

            try:
                winsound.Beep(1800, 500)
            except Exception:
                break

            time.sleep(0.1)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        height, width, _ = image.shape
        status = "Face Not Detected"
        color = (0, 255, 255)
        average_ear = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            average_ear = (left_ear + right_ear) / 2
            
            for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                for index in eye_indices:
                    x = int(landmarks[index].x * width)
                    y = int(landmarks[index].y * height)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(
                image,
                f"EAR: {average_ear:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if average_ear < EAR_THRESHOLD:
                status = "Eyes Closed"
                color = (0, 165, 255)

                if self.closed_since is None:
                    self.closed_since = time.time()

                duration = time.time() - self.closed_since

                if duration >= CLOSED_EYE_SECONDS:
                    status = "DROWSINESS ALERT!"
                    color = (0, 0, 255)

                    if not self.alert_recorded:
                        self.shared_state.add_alert()
                        self.alert_recorded = True

                    with self.lock:
                        if not self.alarm_active:
                            self.alarm_active = True
                            threading.Thread(
                                target=self.play_alarm,
                                daemon=True
                            ).start()

            else:
                self.closed_since = None
                self.alert_recorded = False

                with self.lock:
                    self.alarm_active = False

                status = "Eyes Open"
                color = (0, 255, 0)

        else:
            self.closed_since = None
            self.alert_recorded = False

            with self.lock:
                self.alarm_active = False

        self.shared_state.update_status(status, average_ear)

        cv2.putText(
            image,
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )



# -------------------- HOME PAGE --------------------

if page == "🏠 Home":

    st.title("🚗 Driver Drowsiness Detection")
    st.subheader("Welcome to DrowsyGuard! 👋")

    st.write(
        "An AI-based system that monitors eye closure "
        "and alerts when prolonged eye closure is detected."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #312e81, #1e293b);
            padding: 25px;
            border-radius: 18px;
            border: 1px solid #8b5cf6;
            min-height: 210px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(139,92,246,0.2);
        ">
            <div style="font-size: 45px;">👁️</div>
            <h3 style="color:#c4b5fd;">Eye Monitoring</h3>
            <p style="color:white;">
                Monitors eye movements using AI and MediaPipe
                to identify prolonged eye closure.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #164e63, #1e293b);
            padding: 25px;
            border-radius: 18px;
            border: 1px solid #22d3ee;
            min-height: 210px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(34,211,238,0.15);
        ">
            <div style="font-size: 45px;">🔊</div>
            <h3 style="color:#67e8f9;">Audio Alarm</h3>
            <p style="color:white;">
                Plays an alarm when the system detects
                prolonged eye closure.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #581c87, #1e293b);
            padding: 25px;
            border-radius: 18px;
            border: 1px solid #c084fc;
            min-height: 210px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(192,132,252,0.15);
        ">
            <div style="font-size: 45px;">📊</div>
            <h3 style="color:#d8b4fe;">Alert History</h3>
            <p style="color:white;">
                Keeps a record of detected drowsiness alerts
                and lets you download a CSV report.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("Select Dashboard from the sidebar to start monitoring.")




# -------------------- DASHBOARD --------------------

elif page == "📊 Dashboard":

    st.title("📊 Live Monitoring Dashboard")

    state = st.session_state.detection_state
    alert_count, history, status, ear = state.get_data()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Alerts", alert_count)
    col2.metric("Eye Status", status)
    col3.metric(
        "EAR Value",
        f"{ear:.2f}" if ear is not None else "N/A"
    )

    st.markdown("---")
    st.subheader("📷 Live Camera Monitoring")

    st.checkbox(
        "▶️ Start Monitoring",
        key="monitoring"
    )

    if st.session_state.monitoring:
        webrtc_streamer(
            key="drowsiness-camera",
            video_processor_factory=lambda: DrowsinessDetector(state),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )
    else:
        st.warning("Monitoring is stopped. Tick the checkbox to start.")

    st.markdown("---")
    st.subheader("📈 Session Summary")
    st.write(f"Total drowsiness alerts: **{alert_count}**")

    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("No drowsiness alerts recorded yet.")

    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    st.caption(
        "Use Refresh Dashboard to reload the displayed counter and status."
    )

    st.warning(
        "Educational prototype only. Never rely on this system "
        "instead of proper rest or safe-driving practices."
    )


# -------------------- DETECTION HISTORY --------------------

elif page == "📜 Detection History":

    st.title("📜 Detection History")

    alert_count, history, status, ear = (
        st.session_state.detection_state.get_data()
    )

    st.metric("Total Drowsiness Alerts", alert_count)

    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)

        output = io.StringIO()

        writer = csv.DictWriter(
            output,
            fieldnames=["Alert No.", "Date & Time", "Event"]
        )

        writer.writeheader()
        writer.writerows(history)

        st.download_button(
            "📥 Download CSV Report",
            data=output.getvalue(),
            file_name="drowsyguard_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No alerts yet. Start monitoring from the Dashboard.")


# -------------------- ABOUT PROJECT --------------------

elif page == "🤖 About Project":

    st.title("🤖 About DrowsyGuard")

    st.write(
        "DrowsyGuard is a computer-vision project designed "
        "to demonstrate detection of prolonged eye closure."
    )

    st.subheader("🧠 Technologies Used")

    st.markdown("""
    - **Python:** Main programming language.
    - **OpenCV:** Processes webcam frames.
    - **MediaPipe:** Detects facial landmarks.
    - **NumPy:** Performs numerical calculations.
    - **Streamlit:** Creates the web interface.
    - **Streamlit-WebRTC:** Streams webcam video.
    """)

    st.subheader("⚙️ How It Works")

    st.markdown("""
    1. The webcam captures video.
    2. MediaPipe detects facial landmarks.
    3. The program calculates the Eye Aspect Ratio (EAR).
    4. Prolonged eye closure triggers a drowsiness alert.
    5. The laptop alarm sounds and the event is recorded.
    """)

    st.info(
        "This is an educational prototype, not a certified "
        "driver-safety device."
    )


# -------------------- SAFETY TIPS --------------------

elif page == "🛡️ Safety Tips":

    st.title("🛡️ Driver Safety Tips")

    st.subheader("😴 1. Get Enough Sleep")
    st.write("Rest properly before driving.")

    st.subheader("☕ 2. Take Breaks")
    st.write("Stop safely and rest during long journeys.")

    st.subheader("🚫 3. Do Not Ignore Warning Signs")
    st.write("Heavy eyelids and repeated yawning are warning signs.")

    st.subheader("💊 4. Check Medication Warnings")
    st.write("Some medicines can cause drowsiness.")

    st.subheader("🌙 5. Be Careful at Night")
    st.write("Avoid driving during your usual sleeping hours.")

    st.subheader("🚗 6. Stop Somewhere Safe")
    st.write("If you feel sleepy, stop driving and rest.")

    st.error(
        "This project is an educational prototype. "
        "It can miss drowsiness or produce false alerts. "
        "Never continue driving when sleepy."
    )


# -------------------- FOOTER --------------------

st.markdown("---")

st.markdown(
    "<p style='text-align:center;'>"
    "DrowsyGuard | Python • OpenCV • MediaPipe • Streamlit"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown("""
<div style="
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #312e81, #1e293b);
    border-radius: 15px;
    border: 1px solid #8b5cf6;
    margin-top: 20px;
">
    <h3 style="color: #c4b5fd;">🚗 DrowsyGuard</h3>
    <p style="color: white; font-size: 16px;">
        Developed with ❤️ by <b>Aditi Singh</b>
    </p>
    <p style="color: #cbd5e1; font-size: 13px;">
        Python • OpenCV • MediaPipe • Streamlit
    </p>
    <p style="color: #94a3b8; font-size: 12px;">
        Driver Drowsiness Detection System
    </p>
</div>
""", unsafe_allow_html=True)