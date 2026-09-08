import os
import warnings
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile
from datetime import datetime
import joblib
import time
import atexit
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel, gabor

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fabric Defect Detection System",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .result-good {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .result-defect {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FEATURE EXTRACTION - 89 FIXED FEATURES (MUST MATCH TRAINING)
# =========================
def extract_features_89(image):
    """Extract exactly 89 features - MUST match training"""
    features = []
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. Basic statistics (9)
    features.extend([
        np.mean(gray), np.std(gray), np.median(gray),
        np.percentile(gray, 25), np.percentile(gray, 75),
        np.percentile(gray, 90), np.percentile(gray, 10),
        np.max(gray) - np.min(gray), np.var(gray)
    ])
    
    # 2. Histogram (16)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    features.extend(hist.flatten())
    
    # 3. LBP (16)
    for radius, n_points in [(1, 8), (2, 16)]:
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        hist_padded = np.zeros(8)
        hist_padded[:min(8, len(hist))] = hist[:8]
        features.extend(hist_padded)
    
    # 4. GLCM (8)
    try:
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2], 256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
            prop_vals = graycoprops(glcm, prop)
            features.append(np.mean(prop_vals))
            features.append(np.std(prop_vals))
    except:
        features.extend([0] * 8)
    
    # 5. Edge (6)
    edges_sobel = sobel(gray)
    edges_canny = cv2.Canny(gray, 50, 150)
    features.extend([
        np.mean(edges_sobel), np.std(edges_sobel), np.max(edges_sobel),
        np.sum(edges_canny > 0) / (gray.shape[0] * gray.shape[1]),
        np.percentile(edges_sobel, 75), np.percentile(edges_sobel, 90)
    ])
    
    # 6. Gabor (12)
    gabor_features = []
    try:
        for theta in [0, np.pi/4]:
            for frequency in [0.1, 0.2, 0.3]:
                gabor_real, _ = gabor(gray, frequency=frequency, theta=theta)
                gabor_features.append(np.mean(gabor_real))
                gabor_features.append(np.std(gabor_real))
    except:
        pass
    while len(gabor_features) < 12:
        gabor_features.append(0)
    features.extend(gabor_features[:12])
    
    # 7. Defect features (10)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(thresh1, thresh2)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        features.extend([
            min(len(contours), 5), np.mean(areas) if areas else 0,
            np.std(areas) if len(areas) > 1 else 0, np.max(areas) if areas else 0,
            np.sum(areas), np.sum(areas) / (gray.shape[0] * gray.shape[1])
        ])
        features.extend([0, 0, 0, 0])
    else:
        features.extend([0] * 10)
    
    # 8. Color (12)
    if len(image.shape) == 3:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for channel in range(3):
                features.extend([
                    np.mean(hsv[:,:,channel]), np.std(hsv[:,:,channel]),
                    np.median(hsv[:,:,channel]), np.percentile(hsv[:,:,channel], 25)
                ])
        except:
            features.extend([0] * 12)
    else:
        features.extend([0] * 12)
    
    return np.array(features, dtype=np.float32)

# =========================
# DETECTOR CLASS
# =========================
class FabricDefectDetector:
    def __init__(self):
        self.model_loaded = False
        self.binary_model = None
        self.severity_model = None
        self.scaler = None
        
        try:
            with st.spinner("Loading models..."):
                self.binary_model = joblib.load('binary_classifier.pkl')
                self.severity_model = joblib.load('severity_classifier.pkl')
                self.scaler = joblib.load('feature_scaler.pkl')
                self.model_loaded = True
        except Exception as e:
            st.error(f"❌ Model loading error: {str(e)}")
            st.info("Please run training first: python train_complete.py")
    
    def predict(self, image):
        if not self.model_loaded:
            return "Model not loaded", 0.0, None
        
        try:
            # Preprocess
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            image = cv2.resize(image, (128, 128))
            features = extract_features_89(image)
            features = np.nan_to_num(features.reshape(1, -1))
            
            # Check feature dimension
            if features.shape[1] != 89:
                st.error(f"Feature dimension mismatch! Expected 89, got {features.shape[1]}")
                return "Error", 0.0, None
            
            features_scaled = self.scaler.transform(features)
            
            # Binary prediction
            pred = self.binary_model.predict(features_scaled)[0]
            prob = self.binary_model.predict_proba(features_scaled)[0]
            confidence = max(prob)
            
            if pred == 0:
                return "Good", confidence, None
            else:
                if self.severity_model is not None:
                    sev_pred = self.severity_model.predict(features_scaled)[0]
                    severity_labels = ["Low", "Medium", "High"]
                    return f"Defect - {severity_labels[sev_pred]}", confidence, severity_labels[sev_pred]
                else:
                    return "Defect", confidence, None
        except Exception as e:
            return f"Error: {str(e)[:50]}", 0.0, None

# =========================
# INITIALIZE SESSION STATE
# =========================
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# Cleanup function
def cleanup_temp_files():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass

atexit.register(cleanup_temp_files)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://via.placeholder.com/150x150?text=Fabric+AI", width=150)
    st.title("🎯 Controls")
    
    if st.button("📥 Load Detection Model", use_container_width=True):
        st.session_state.detector = FabricDefectDetector()
        if st.session_state.detector and st.session_state.detector.model_loaded:
            st.success("✅ Models loaded successfully!")
    
    st.markdown("---")
    
    if st.session_state.detector and st.session_state.detector.model_loaded:
        st.success("✅ Model Status: Active")
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Lower values = more sensitive detection"
        )
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.detection_history:
            total = len(st.session_state.detection_history)
            defects = sum(1 for r in st.session_state.detection_history if "Defect" in r.get('result', ''))
            st.metric("Total Detections", total)
            st.metric("Defects Found", defects)
            if total > 0:
                st.metric("Defect Rate", f"{(defects/total)*100:.1f}%")
        
        st.markdown("---")
        st.info("""
        **Features:**
        - 📸 Image Analysis
        - 🎥 Video Analysis
        - 📹 Live Camera
        - 📊 Analytics Dashboard
        - 📜 History Tracking
        """)
    else:
        st.warning("⚠️ Model not loaded")

# =========================
# MAIN TABS
# =========================
st.title("🧵 Fabric Defect Detection System")
st.markdown("*AI-powered quality control for textile manufacturing*")

if st.session_state.detector is None or not st.session_state.detector.model_loaded:
    st.warning("⚠️ Please load the detection model using the sidebar button")
    st.info("""
    ### Getting Started:
    1. Click 'Load Detection Model' in the sidebar
    2. Choose from Image, Video, or Camera tab
    3. Get instant defect detection results
    
    ### Features:
    - **Image Analysis**: Upload single images
    - **Video Analysis**: Process video files
    - **Live Camera**: Real-time detection
    - **Severity Assessment**: Low/Medium/High
    - **Confidence Scoring**: Probability of detection
    - **Detection History**: Track all analyses
    """)
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Image Analysis",
    "🎥 Video Analysis",
    "📹 Live Camera",
    "📊 Analytics Dashboard",
    "📜 Detection History"
])

# =========================
# TAB 1: IMAGE ANALYSIS
# =========================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Fabric Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fabric", width="stretch")
            
            if st.button("🔍 Analyze Image", key="analyze_image", use_container_width=True):
                with st.spinner("Analyzing fabric quality..."):
                    img_array = np.array(image)
                    result, confidence, severity = st.session_state.detector.predict(img_array)
                    
                    # Store in history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'image',
                        'result': result,
                        'confidence': confidence,
                        'severity': severity,
                        'filename': uploaded_file.name
                    })
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        if "Good" in result:
                            st.markdown('<div class="result-good">', unsafe_allow_html=True)
                            st.success(f"### ✅ {result}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-defect">', unsafe_allow_html=True)
                            st.error(f"### ⚠️ {result}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                        if severity:
                            st.subheader("Severity Level")
                            severity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[severity]
                            st.markdown(f"### {severity_color} {severity}")
                            
                            # Severity meter
                            severity_value = {"Low": 0.33, "Medium": 0.66, "High": 1.0}[severity]
                            st.progress(severity_value)
                        
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={'text': "Confidence"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "darkgray"}
                                ]
                            }
                        ))
                        fig.update_layout(height=200)
                        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: VIDEO ANALYSIS
# =========================
with tab2:
    st.subheader("Video File Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video_upload"
    )
    
    if uploaded_video:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name
        st.session_state.temp_files.append(video_path)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
            
            if st.button("🎬 Analyze Video", key="analyze_video", use_container_width=True):
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    frame_results = []
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every 10th frame for efficiency
                        if frame_count % 10 == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result, confidence, severity = st.session_state.detector.predict(frame_rgb)
                            
                            frame_results.append({
                                'result': result,
                                'confidence': confidence,
                                'severity': severity
                            })
                            
                            # Update progress
                            if total_frames > 0:
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                        
                        frame_count += 1
                    
                    cap.release()
                    
                    # Calculate statistics
                    total_processed = len(frame_results)
                    defect_frames = sum(1 for r in frame_results if "Defect" in r['result'])
                    avg_confidence = np.mean([r['confidence'] for r in frame_results]) if frame_results else 0
                    
                    # Store in history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'video',
                        'result': f"Video: {defect_frames}/{total_processed} defects",
                        'confidence': avg_confidence,
                        'severity': None,
                        'filename': uploaded_video.name
                    })
                    
                    with col2:
                        st.subheader("Video Analysis Results")
                        st.metric("Frames Analyzed", total_processed)
                        st.metric("Defect Frames", defect_frames)
                        st.metric("Defect Rate", f"{(defect_frames/total_processed)*100:.1f}%" if total_processed > 0 else "0%")
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Severity distribution
                        severities = [r['severity'] for r in frame_results if r['severity']]
                        if severities:
                            st.subheader("Severity Distribution")
                            sev_counts = pd.Series(severities).value_counts().reset_index()
                            sev_counts.columns = ['Severity', 'Count']
                            fig = px.pie(sev_counts, values='Count', names='Severity', title="Defect Severity")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Video analysis complete!")

# =========================
# TAB 3: LIVE CAMERA
# =========================
with tab3:
    st.subheader("Real-time Camera Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_placeholder = st.empty()
        
        col_buttons = st.columns(2)
        with col_buttons[0]:
            start_camera = st.button("📷 Start Camera", key="start_cam", use_container_width=True)
        with col_buttons[1]:
            stop_camera = st.button("⏹️ Stop Camera", key="stop_cam", use_container_width=True)
        
        if start_camera:
            st.session_state.camera_active = True
        
        if stop_camera:
            st.session_state.camera_active = False
        
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access camera. Please check your camera connection.")
                st.session_state.camera_active = False
            else:
                fps_display = st.empty()
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result, confidence, severity = st.session_state.detector.predict(frame_rgb)
                    
                    # Add overlay
                    if "Defect" in result:
                        color = (0, 0, 255)
                        status_text_cam = f"⚠️ {result}"
                    else:
                        color = (0, 255, 0)
                        status_text_cam = f"✅ {result}"
                    
                    # Draw on frame
                    cv2.putText(frame, status_text_cam, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if severity:
                        cv2.putText(frame, f"Severity: {severity}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        fps_display.info(f"📊 FPS: {fps:.1f}")
                    
                    # Display
                    camera_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Store every 30 frames
                    if frame_count % 30 == 0:
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'type': 'camera',
                            'result': result,
                            'confidence': confidence,
                            'severity': severity,
                            'filename': 'camera_feed'
                        })
                    
                    time.sleep(0.03)
                
                cap.release()
                st.session_state.camera_active = False
    
    with col2:
        st.subheader("Live Statistics")
        
        # Show recent camera detections
        camera_detections = [r for r in st.session_state.detection_history if r['type'] == 'camera']
        if camera_detections:
            recent = camera_detections[-5:]
            st.write("**Recent Detections:**")
            for det in reversed(recent):
                timestamp = det['timestamp'].strftime("%H:%M:%S")
                result = det['result']
                confidence = det['confidence']
                if "Defect" in result:
                    st.warning(f"{timestamp}: {result} ({confidence:.1%})")
                else:
                    st.success(f"{timestamp}: {result} ({confidence:.1%})")
        else:
            st.info("No camera detections yet. Start the camera to begin.")

# =========================
# TAB 4: ANALYTICS DASHBOARD
# =========================
with tab4:
    st.subheader("Detection Analytics Dashboard")
    
    if st.session_state.detection_history:
        # Create DataFrame
        df_data = []
        for record in st.session_state.detection_history:
            df_data.append({
                'timestamp': record['timestamp'],
                'type': record['type'],
                'result': record['result'],
                'confidence': record['confidence'],
                'severity': record.get('severity', 'N/A')
            })
        
        df = pd.DataFrame(df_data)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Detections", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            defects = len(df[df['result'].str.contains("Defect", na=False)])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Defects Found", defects)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            avg_conf = df['confidence'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            defect_rate = (defects / len(df)) * 100 if len(df) > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Defect Rate", f"{defect_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Time series
        st.subheader("Confidence Trend Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['confidence'],
                                 mode='lines+markers', name='Confidence',
                                 line=dict(color='#667eea', width=2)))
        fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Confidence")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Result Distribution")
            result_counts = df['result'].value_counts().reset_index()
            result_counts.columns = ['Result', 'Count']
            fig_pie = px.pie(result_counts, values='Count', names='Result',
                            title="Detection Results",
                            color_discrete_sequence=['#84fab0', '#f5576c'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Input Type Distribution")
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig_bar = px.bar(type_counts, x='Type', y='Count',
                            title="Detections by Input Type",
                            color='Type',
                            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Severity analysis
        severity_df = df[df['severity'] != 'N/A']
        if not severity_df.empty:
            st.subheader("Severity Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                sev_counts = severity_df['severity'].value_counts().reset_index()
                sev_counts.columns = ['Severity', 'Count']
                fig_sev = px.bar(sev_counts, x='Severity', y='Count',
                                title="Severity Distribution",
                                color='Severity',
                                color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'})
                st.plotly_chart(fig_sev, use_container_width=True)
            
            with col2:
                sev_by_type = pd.crosstab(severity_df['type'], severity_df['severity'])
                fig_heatmap = px.imshow(sev_by_type.values,
                                       x=sev_by_type.columns,
                                       y=sev_by_type.index,
                                       title="Severity by Input Type",
                                       color_continuous_scale='Viridis',
                                       aspect='auto')
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No data available. Upload images, videos, or use camera to see analytics.")

# =========================
# TAB 5: DETECTION HISTORY
# =========================
with tab5:
    st.subheader("Detection History")
    
    if st.session_state.detection_history:
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export to CSV", use_container_width=True):
                export_data = []
                for record in st.session_state.detection_history:
                    export_data.append({
                        'Timestamp': record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'Type': record['type'],
                        'Result': record['result'],
                        'Confidence': f"{record['confidence']:.2%}",
                        'Severity': record.get('severity', 'N/A'),
                        'Filename': record.get('filename', 'N/A')
                    })
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                st.download_button("Download CSV", csv, 
                                  f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                  "text/csv", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.detection_history = []
                st.rerun()
        
        with col3:
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation feature coming soon!")
        
        # Display history table
        display_data = []
        for i, record in enumerate(reversed(st.session_state.detection_history[-50:])):
            display_data.append({
                '#': len(st.session_state.detection_history) - i,
                'Time': record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Type': record['type'].upper(),
                'Result': record['result'],
                'Confidence': f"{record['confidence']:.1%}",
                'Severity': record.get('severity', '-')
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        total = len(st.session_state.detection_history)
        defects = sum(1 for r in st.session_state.detection_history if "Defect" in r['result'])
        images = sum(1 for r in st.session_state.detection_history if r['type'] == 'image')
        videos = sum(1 for r in st.session_state.detection_history if r['type'] == 'video')
        cameras = sum(1 for r in st.session_state.detection_history if r['type'] == 'camera')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", total)
        with col2:
            st.metric("Defects Found", defects)
        with col3:
            quality_rate = ((total - defects) / total) * 100 if total > 0 else 100
            st.metric("Quality Rate", f"{quality_rate:.1f}%")
        with col4:
            st.metric("Unique Files", len(set(r.get('filename', '') for r in st.session_state.detection_history)))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📸 Images: {images}")
        with col2:
            st.info(f"🎥 Videos: {videos}")
        with col3:
            st.info(f"📹 Camera: {cameras}")
    else:
        st.info("No detection history available. Upload images, videos, or use camera to build history.")
        
        # Show getting started guide
        st.markdown("""
        ### 📋 Getting Started Guide
        
        **Step 1:** Load the detection model using the sidebar button
        
        **Step 2:** Choose your input method:
        - **Image Analysis**: Upload single images for defect detection
        - **Video Analysis**: Upload video files for batch processing
        - **Live Camera**: Real-time detection with webcam
        
        **Step 3:** View results in the Analytics Dashboard
        
        **Step 4:** Export history for reporting
        
        ### 💡 Tips for Best Results
        - Ensure good lighting conditions
        - Keep camera steady for video analysis
        - Use high-resolution images when possible
        - Adjust confidence threshold in sidebar if needed
        """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🧵 Fabric Defect Detection System | Powered by Random Forest Classifier</p>
        <p style='font-size: 12px;'>89-Dimensional Feature Extraction | Real-time Quality Control</p>
    </div>
    """,
    unsafe_allow_html=True
)
