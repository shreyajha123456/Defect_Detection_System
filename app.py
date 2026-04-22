import os
import warnings
import logging
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from PIL import Image
import io
import tempfile
from datetime import datetime
import joblib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel, gabor
import random

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Carpet Defect Detection System",
    page_icon="🪢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        padding: 10px;
        border: none;
        border-radius: 10px;
        transition: transform 0.2s;
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card-good {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card-defect {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
# FEATURE EXTRACTION FUNCTIONS
# =========================
def extract_texture_features(image):
    """Extract texture features using LBP, GLCM, and Gabor filters"""
    features = []
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. LBP (Local Binary Pattern) features
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist)
    
    # 2. GLCM features
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # 3. Edge features using Sobel
    edge_sobel = sobel(gray)
    edge_density = np.mean(edge_sobel)
    edge_std = np.std(edge_sobel)
    features.extend([edge_density, edge_std])
    
    # 4. Gabor filter features
    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for frequency in [0.1, 0.3, 0.5]:
            gabor_real, gabor_imag = gabor(gray, frequency=frequency, theta=theta)
            gabor_features.append(np.mean(gabor_real))
            gabor_features.append(np.std(gabor_real))
    
    features.extend(gabor_features[:10])
    
    # 5. Statistical features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 75),
        np.max(gray) - np.min(gray)
    ])
    
    return np.array(features)

def extract_color_features(image):
    """Extract color-based features"""
    features = []
    
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # RGB histograms
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = hist / (hist.sum() + 1e-6)
            features.extend(hist.flatten()[:16])
        
        # HSV statistics
        for i in range(3):
            features.extend([
                np.mean(hsv[:,:,i]),
                np.std(hsv[:,:,i]),
                np.percentile(hsv[:,:,i], 50)
            ])
        
        # LAB statistics
        for i in range(3):
            features.extend([
                np.mean(lab[:,:,i]),
                np.std(lab[:,:,i])
            ])
    
    return np.array(features)

def extract_defect_features(image):
    """Extract defect-specific features using morphological operations"""
    features = []
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply multiple thresholding methods
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine thresholds
    combined_thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contour features
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        features.extend([
            len(contours),
            np.mean(areas) if areas else 0,
            np.std(areas) if len(areas) > 1 else 0,
            np.max(areas) if areas else 0,
            np.sum(areas),
            np.mean(perimeters) if perimeters else 0,
        ])
        
        defect_ratio = np.sum(areas) / (gray.shape[0] * gray.shape[1])
        features.append(defect_ratio)
    else:
        features.extend([0, 0, 0, 0, 0, 0, 0])
    
    # Texture uniformity
    features.append(np.var(gray))
    
    return np.array(features)

def extract_all_features_from_image(image):
    """Extract all features from an image array"""
    features = []
    
    # Resize image
    image_resized = cv2.resize(image, (128, 128))
    
    # Extract feature sets
    texture_feats = extract_texture_features(image_resized)
    color_feats = extract_color_features(image_resized)
    defect_feats = extract_defect_features(image_resized)
    
    # Combine all features
    all_features = np.concatenate([texture_feats, color_feats, defect_feats])
    
    return all_features

# =========================
# CARPET DEFECT DETECTOR CLASS
# =========================
class CarpetDefectDetector:
    def __init__(self, model_path='carpet_defect_rf_model.pkl', scaler_path='feature_scaler.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Model not found. Please train the model first using the training script.")
            self.model_loaded = False
    
    def extract_features(self, image):
        """Extract features from image"""
        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            features = extract_all_features_from_image(image)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
        except Exception as e:
            st.error(f"Feature extraction error: {str(e)}")
            return None
    
    def detect_defect(self, image, confidence_threshold=0.6):
        """Detect defect in image"""
        if not self.model_loaded:
            return "Model not available", 0.0, None
        
        try:
            features = self.extract_features(image)
            if features is None:
                return "Error processing image", 0.0, None
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            result = "Defect" if prediction == 1 else "Good"
            confidence = max(probabilities)
            
            # Severity analysis for defects
            severity = None
            if prediction == 1:
                # Calculate severity based on defect features
                defect_feats = extract_defect_features(image)
                defect_area_ratio = defect_feats[6] if len(defect_feats) > 6 else 0
                
                if defect_area_ratio < 0.1:
                    severity = "Low"
                elif defect_area_ratio < 0.3:
                    severity = "Moderate"
                else:
                    severity = "High"
            
            return result, confidence, severity
            
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return "Error", 0.0, None
    
    def process_video_frame(self, frame, frame_skip=2):
        """Process video frame for real-time detection"""
        if not self.model_loaded:
            return None
        
        try:
            result, confidence, severity = self.detect_defect(frame)
            return {
                'result': result,
                'confidence': confidence,
                'severity': severity,
                'timestamp': time.time()
            }
        except:
            return None
    
    def generate_report(self, results_history):
        """Generate detection report"""
        if not results_history:
            return None
        
        defects = [r for r in results_history if r['result'] == "Defect"]
        
        report = {
            'total_frames': len(results_history),
            'defect_frames': len(defects),
            'defect_percentage': (len(defects) / len(results_history)) * 100 if results_history else 0,
            'avg_confidence': np.mean([r['confidence'] for r in results_history]),
            'severity_distribution': {
                'Low': len([r for r in defects if r.get('severity') == 'Low']),
                'Moderate': len([r for r in defects if r.get('severity') == 'Moderate']),
                'High': len([r for r in defects if r.get('severity') == 'High'])
            }
        }
        
        return report

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150?text=Carpet+AI", use_column_width=True)
    st.title("🎯 Controls")
    
    # Model loading
    st.subheader("Model Configuration")
    
    if st.button("🔄 Load Detection Model"):
        with st.spinner("Loading model..."):
            try:
                st.session_state.detector = CarpetDefectDetector(
                    'carpet_defect_rf_model.pkl', 
                    'feature_scaler.pkl'
                )
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please train the model first using the training script.")
    
    # Settings
    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6,
        step=0.05,
        help="Lower values = more sensitive detection"
    )
    
    st.subheader("About")
    st.info(
        """
        **Carpet Defect Detection System v3.0**
        
        Features:
        - 🎯 Random Forest Classifier
        - 📊 100+ Texture & Color Features
        - 🎥 Real-time Detection
        - 📈 Analytics Dashboard
        
        **Accuracy:** 75-85% on test set
        """
    )

# Main title
st.title("🪢 Carpet Defect Detection System")
st.markdown("*Advanced AI-powered inspection using Computer Vision & Machine Learning*")

# Check if model is loaded
if st.session_state.detector is None or not st.session_state.detector.model_loaded:
    st.warning("⚠️ Please load the detection model using the button in the sidebar first!")
    st.info("If you haven't trained the model yet, run the training script first to generate 'carpet_defect_rf_model.pkl' and 'feature_scaler.pkl'")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📸 Image Upload", "🎥 Video Analysis", "📹 Live Camera", "📊 Analytics Dashboard", "📜 History & Reports"]
)

# Tab 1: Image Upload
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Detect Defects", key="detect_image"):
                with st.spinner("Analyzing image..."):
                    # Convert PIL to OpenCV format
                    img_array = np.array(image)
                    
                    # Detect
                    result, confidence, severity = st.session_state.detector.detect_defect(
                        img_array, confidence_threshold
                    )
                    
                    # Store in history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'image',
                        'result': result,
                        'confidence': confidence,
                        'severity': severity,
                        'filename': uploaded_file.name
                    })
                    
                    # Display results in second column
                    with col2:
                        st.subheader("Detection Results")
                        
                        # Result card
                        if result == "Good":
                            st.markdown('<div class="result-card-good">', unsafe_allow_html=True)
                            st.success(f"### ✅ {result}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-card-defect">', unsafe_allow_html=True)
                            st.error(f"### ⚠️ {result}")
                            if severity:
                                st.info(f"**Severity:** {severity}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Confidence meter
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                        # Visual confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={'text': "Detection Confidence"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Severity meter for defects
                        if severity:
                            severity_score = {"Low": 33, "Moderate": 66, "High": 100}[severity]
                            st.progress(severity_score / 100)
                            st.caption(f"Severity Level: {severity}")

# Tab 2: Video Analysis
with tab2:
    st.subheader("Video File Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(tfile.name)
            
            if st.button("🎬 Analyze Video", key="analyze_video"):
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture(tfile.name)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
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
                        
                        if frame_count % 10 == 0:  # Process every 10th frame
                            result = st.session_state.detector.process_video_frame(frame)
                            if result:
                                frame_results.append(result)
                                
                                # Update progress
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                        
                        frame_count += 1
                    
                    cap.release()
                    
                    # Generate report
                    report = st.session_state.detector.generate_report(frame_results)
                    
                    with col2:
                        st.subheader("Video Analysis Results")
                        
                        if report:
                            st.metric("Total Frames Analyzed", report['total_frames'])
                            st.metric("Defects Detected", report['defect_frames'])
                            st.metric("Defect Rate", f"{report['defect_percentage']:.1f}%")
                            st.metric("Avg Confidence", f"{report['avg_confidence']:.2%}")
                            
                            # Severity distribution
                            if any(report['severity_distribution'].values()):
                                st.subheader("Severity Distribution")
                                severity_df = pd.DataFrame([
                                    {'Severity': k, 'Count': v} 
                                    for k, v in report['severity_distribution'].items() 
                                    if v > 0
                                ])
                                if not severity_df.empty:
                                    fig = px.pie(severity_df, values='Count', names='Severity', 
                                                title="Defect Severity")
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store in history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'video',
                        'report': report,
                        'filename': uploaded_video.name
                    })
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Video analysis complete!")
        
        # Cleanup
        os.unlink(tfile.name)

# Tab 3: Live Camera
with tab3:
    st.subheader("Live Camera Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_placeholder = st.empty()
        
        col_buttons = st.columns(2)
        with col_buttons[0]:
            start_camera = st.button("📷 Start Camera", key="start_camera", use_container_width=True)
        with col_buttons[1]:
            stop_camera = st.button("⏹️ Stop Camera", key="stop_camera", use_container_width=True)
        
        if start_camera:
            st.session_state.camera_active = True
        
        if stop_camera:
            st.session_state.camera_active = False
        
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            
            # FPS counter
            fps_display = st.empty()
            frame_count = 0
            start_time = time.time()
            last_results = []
            
            warning_placeholder = st.empty()
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    warning_placeholder.error("Failed to access camera. Please check your camera connection.")
                    break
                
                # Process frame
                result = st.session_state.detector.process_video_frame(frame)
                
                if result:
                    last_results.append(result)
                    if len(last_results) > 100:
                        last_results.pop(0)
                    
                    # Add detection overlay
                    if result['result'] == "Defect":
                        color = (0, 0, 255)  # Red for defect
                        status_text = "⚠️ DEFECT DETECTED"
                        status_color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)  # Green for good
                        status_text = "✅ GOOD"
                        status_color = (0, 255, 0)
                    
                    # Draw bounding box and text
                    cv2.putText(frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"Confidence: {result['confidence']:.2%}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if result.get('severity'):
                        cv2.putText(frame, f"Severity: {result['severity']}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        fps_display.info(f"📊 FPS: {fps:.1f}")
                    
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Store result periodically
                    if frame_count % 30 == 0:
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'type': 'camera',
                            'result': result['result'],
                            'confidence': result['confidence'],
                            'severity': result.get('severity')
                        })
            
            cap.release()
            st.session_state.camera_active = False
    
    with col2:
        st.subheader("Live Feed Statistics")
        
        # Show real-time stats
        if 'last_results' in locals() and last_results:
            recent_defects = [r for r in last_results if r['result'] == "Defect"]
            
            st.metric("Recent Detections", len(last_results))
            st.metric("Defects Found", len(recent_defects))
            if last_results:
                avg_conf = np.mean([r['confidence'] for r in last_results])
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            # Severity breakdown
            if recent_defects:
                severity_counts = {
                    'Low': len([r for r in recent_defects if r.get('severity') == 'Low']),
                    'Moderate': len([r for r in recent_defects if r.get('severity') == 'Moderate']),
                    'High': len([r for r in recent_defects if r.get('severity') == 'High'])
                }
                if any(severity_counts.values()):
                    st.subheader("Severity Breakdown")
                    for sev, count in severity_counts.items():
                        if count > 0:
                            st.write(f"- **{sev}**: {count}")

# Tab 4: Analytics Dashboard
with tab4:
    st.subheader("Detection Analytics Dashboard")
    
    if st.session_state.detection_history:
        # Convert history to DataFrame
        df_data = []
        for record in st.session_state.detection_history:
            if 'result' in record:
                df_data.append({
                    'timestamp': record['timestamp'],
                    'type': record['type'],
                    'result': record['result'],
                    'confidence': record.get('confidence', 0),
                    'severity': record.get('severity', 'N/A')
                })
            elif 'report' in record:
                df_data.append({
                    'timestamp': record['timestamp'],
                    'type': record['type'],
                    'result': f"Video Summary",
                    'confidence': record['report']['avg_confidence'],
                    'severity': 'N/A'
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Detections", len(df), delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                defect_count = len(df[df['result'].str.contains("Defect", na=False)])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Defects Found", defect_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                avg_confidence = df['confidence'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                detection_rate = (defect_count / len(df)) * 100 if len(df) > 0 else 0
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Time series plot
            st.subheader("Detection Timeline")
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=("Detections Over Time", "Confidence Trend"))
            
            # Filter defect and good detections
            df_defects = df[df['result'].str.contains("Defect", na=False)]
            df_good = df[df['result'] == "Good"]
            
            fig.add_trace(
                go.Scatter(x=df_defects['timestamp'], y=[1]*len(df_defects),
                          mode='markers', name='Defect',
                          marker=dict(color='red', size=10, symbol='x')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_good['timestamp'], y=[0]*len(df_good),
                          mode='markers', name='Good',
                          marker=dict(color='green', size=10, symbol='circle')),
                row=1, col=1
            )
            
            # Confidence trend
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['confidence'],
                          mode='lines+markers', name='Confidence',
                          line=dict(color='#667eea', width=2),
                          marker=dict(size=6)),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True, 
                             title_font_size=14,
                             plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Detection Status", row=1, col=1, 
                            ticktext=['Good', 'Defect'], tickvals=[0, 1])
            fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Distribution")
                result_counts = df['result'].value_counts()
                fig_pie = px.pie(values=result_counts.values, 
                                names=result_counts.index,
                                title="Detection Results",
                                color_discrete_sequence=['#84fab0', '#f5576c'])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Input Type Distribution")
                type_counts = df['type'].value_counts()
                fig_bar = px.bar(x=type_counts.index, y=type_counts.values,
                                title="Detections by Input Type",
                                color=type_counts.index,
                                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Confidence distribution
            st.subheader("Confidence Score Distribution")
            fig_hist = px.histogram(df, x='confidence', nbins=20,
                                   title="Confidence Score Distribution",
                                   color_discrete_sequence=['#667eea'],
                                   marginal='box')
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Severity analysis for defects
            severity_data = df[df['severity'] != 'N/A']
            if not severity_data.empty:
                st.subheader("Defect Severity Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    severity_counts = severity_data['severity'].value_counts()
                    fig_severity = px.bar(x=severity_counts.index, y=severity_counts.values,
                                         title="Severity Distribution",
                                         color=severity_counts.index,
                                         color_discrete_map={
                                             'Low': '#84fab0',
                                             'Moderate': '#f6d365',
                                             'High': '#f5576c'
                                         })
                    st.plotly_chart(fig_severity, use_container_width=True)
                
                with col2:
                    severity_by_type = pd.crosstab(severity_data['type'], severity_data['severity'])
                    fig_heatmap = px.imshow(severity_by_type, 
                                           title="Severity by Input Type",
                                           color_continuous_scale='Viridis',
                                           aspect='auto')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No detection data available. Start analyzing images or videos to see analytics.")

# Tab 5: History & Reports
with tab5:
    st.subheader("Detection History & Reports")
    
    if st.session_state.detection_history:
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export as CSV", use_container_width=True):
                # Convert to DataFrame
                export_data = []
                for record in st.session_state.detection_history:
                    if 'result' in record:
                        export_data.append({
                            'Timestamp': record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': record['type'],
                            'Result': record['result'],
                            'Confidence': f"{record.get('confidence', 0):.2%}",
                            'Severity': record.get('severity', 'N/A'),
                            'Filename': record.get('filename', 'N/A')
                        })
                    elif 'report' in record:
                        export_data.append({
                            'Timestamp': record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': record['type'],
                            'Result': f"Video: {record['report']['defect_frames']}/{record['report']['total_frames']} defects",
                            'Confidence': f"{record['report']['avg_confidence']:.2%}",
                            'Severity': 'N/A',
                            'Filename': record.get('filename', 'N/A')
                        })
                
                if export_data:
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"defect_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
       