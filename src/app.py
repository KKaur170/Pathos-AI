import streamlit as st
import librosa
import joblib
import pandas as pd
import numpy as np
import torch
import os
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from streamlit_mic_recorder import mic_recorder
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
import soundfile as sf
import io

ffmpeg_bin = r"C:\Users\KHUSHNOOR KAUR\Downloads\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
from pydub import AudioSegment


# --- 0. ADD THIS CLASS DEFINITION ---

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=1000, depth=6, learning_rate=0.05, class_weights=None):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.model = CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            loss_function='MultiClass', random_seed=42, verbose=False,
            class_weights=class_weights
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def get_params(self, deep=True):
        return {"iterations": self.iterations, "depth": self.depth,
                "learning_rate": self.learning_rate, "class_weights": self.class_weights}
    def set_params(self, **params):
        for param, value in params.items(): setattr(self, param, value)
        return self
# ---------------------------------

# --- 1. CONFIGURATION ---
PATH_MLY = "models/malayalam/"
PATH_PNJ = "models/punjabi/"

# --- 2. CACHED LOADING ---
@st.cache_resource
def load_bert_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    return tokenizer, model

@st.cache_resource
def load_assets(language):
    path = PATH_MLY if language == "Malayalam" else PATH_PNJ
    if language == "Malayalam":
        return (joblib.load(os.path.join(path, "voting_model.pkl")),
                joblib.load(os.path.join(path, "scaler.pkl")),
                joblib.load(os.path.join(path, "thresholds.pkl")),
                joblib.load(os.path.join(path, "features_used.pkl")))
# In load_assets, change the Punjabi return:
    else: # Punjabi
        return (
          joblib.load(os.path.join(path, "stacking_model.joblib")),
          joblib.load(os.path.join(path, "scaler.joblib")),
          "median", # Return the string strategy instead of the object
          pd.read_csv(os.path.join(path, "topk_features.csv")).iloc[:, 0].tolist(),
          joblib.load(os.path.join(path, "all_feature_cols.joblib")) # Ensure this is loaded
        )   

tokenizer, text_model = load_bert_models()

# --- 3. EXTRACTION LOGIC ---
import librosa
import numpy as np
import parselmouth # You will need this for F0/Jitter/Shimmer
from parselmouth.praat import call

# --- MALAYALAM EXTRACTION (Multimodal: Audio + BERT) ---
def extract_malayalam_features(path: str, text: str) -> dict:
    # 1. Load Audio using soundfile backend to avoid NoBackendError
    y, sr = librosa.load(path, sr=None, mono=True)
    feats = {}
    
    # Audio features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13): feats[f"mfcc_{i}"] = float(np.mean(mfcc[i]))
    feats["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["rms"] = float(np.mean(librosa.feature.rms(y=y)))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]): feats[f"chroma_{i}"] = float(np.mean(chroma[i]))
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]): feats[f"contrast_{i}"] = float(np.mean(contrast[i]))
    
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(tonnetz.shape[0]): feats[f"tonnetz_{i}"] = float(np.mean(tonnetz[i]))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    for i in range(min(128, mel.shape[0])): feats[f"mel_{i}"] = float(np.mean(mel[i]))
    
    # 2. Text Embeddings
    text_embeddings = get_mean_pooling_embedding(text)
    for i, val in enumerate(text_embeddings):
        feats[f"bert_{i}"] = float(val)
        
    return feats

# --- PUNJABI EXTRACTION (Prosody & Voice Quality) ---
def extract_punjabi_features(path: str) -> dict:
    # Force soundfile backend here too
    y, sr = librosa.load(path, sr=None, mono=True)
    sound = parselmouth.Sound(path)
    feats = {}

    # 1. MFCCs (Mean & Std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f'mfcc_mean_{i}'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_std_{i}'] = float(np.std(mfcc[i]))

    # 2. Spectral Dynamics
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std'] = float(np.std(rms))
    feats['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats['centroid_mean'] = float(np.mean(centroid))
    feats['bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # 3. Chroma & Contrast
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f'chroma_mean_{i}'] = float(np.mean(chroma[i]))
        feats[f'chroma_std_{i}'] = float(np.std(chroma[i]))
        
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f'contrast_mean_{i}'] = float(np.mean(contrast[i]))
        feats[f'contrast_std_{i}'] = float(np.std(contrast[i]))

    # 4. Prosody (Parselmouth)
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        feats['f0_mean'] = float(call(pitch, "Get mean", 0, 0, "Hertz"))
        feats['f0_std'] = float(call(pitch, "Get standard deviation", 0, 0, "Hertz"))
        feats['f0_max'] = float(call(pitch, "Get maximum", 0, 0, "Hertz", "None"))
        feats['f0_min'] = float(call(pitch, "Get minimum", 0, 0, "Hertz", "None"))
        
        dur = librosa.get_duration(y=y, sr=sr)
        feats['duration_sec'] = float(dur)
        feats['f0_slope'] = float((feats['f0_max'] - feats['f0_min']) / dur) if dur > 0 else 0.0
        
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        feats['jitter_local'] = float(call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats['shimmer_local'] = float(call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats['hnr_db'] = float(call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0).GetMean(0, 0))
    except:
        # Fallback values
        for k in ['f0_mean','f0_std','f0_max','f0_min','f0_slope','jitter_local','shimmer_local','hnr_db']:
            feats[k] = 0.0
            
    return feats

def extract_basic_librosa_features(y, sr):
    feats = {}
    # 1. MFCCs (Already there, good)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13): 
        feats[f"mfcc_mean_{i}"] = np.mean(mfcc[i])
        feats[f"mfcc_std_{i}"] = np.std(mfcc[i]) # You need _std too!

    # 2. Spectral & Dynamics
    zcr = librosa.feature.zero_crossing_rate(y)
    feats['zcr_mean'] = np.mean(zcr)
    feats['zcr_std'] = np.std(zcr)
    
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = np.mean(rms)
    feats['rms_std'] = np.std(rms)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats['centroid_mean'] = np.mean(centroid)
    feats['centroid_std'] = np.std(centroid)
    
    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f"chroma_mean_{i}"] = np.mean(chroma[i])
        
    # NOTE: You MUST also include the Parselmouth features 
    # (f0, jitter, shimmer, etc.) in extract_punjabi_features 
    # and add them to the 'feats' dictionary returned to run_prediction.
    
    return feats

def get_mean_pooling_embedding(text: str) -> np.ndarray:
    toks = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad(): out = text_model(**toks).last_hidden_state
    mask = toks["attention_mask"].unsqueeze(-1).expand(out.size()).float()
    summed = torch.sum(out * mask, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return (summed / summed_mask).squeeze().cpu().numpy()


# --- 4. PREDICTION LOGIC (FIXED) ---
def run_prediction(audio_path, text=None):

    import os
    # Inside run_prediction, before extract_punjabi_features:
    if not os.path.exists(audio_path):
        st.error(f"File not found at {audio_path}")
        return None, None, None
    # --- MALAYALAM ---
    if text is not None and str(text).strip():
        model, scaler, thresh_dict, feat_order = load_assets("Malayalam")
        features = extract_malayalam_features(audio_path, text)
        row = [features.get(f, 0.0) for f in feat_order]
        Xs = scaler.transform(np.array([row]))
        probs = model.predict_proba(Xs)[0]
        labels = model.classes_
        
        # Margin logic
        best_idx = apply_margin_logic(probs, labels, thresh_dict)
        return probs, labels, best_idx

    # --- PUNJABI ---
    else:
        model, scaler, strategy, top_features, full_feature_list = load_assets("Punjabi")
        le = joblib.load(os.path.join(PATH_PNJ, "label_encoder.joblib"))
        thresholds = joblib.load(os.path.join(PATH_PNJ, "thresholds.joblib"))
        
        features = extract_punjabi_features(audio_path)
        df_full = pd.DataFrame([{f: features.get(f, 0.0) for f in full_feature_list}]).fillna(0.0)
        
        feat_to_idx = {f: i for i, f in enumerate(full_feature_list)}
        top_indices = [feat_to_idx[f] for f in top_features]
        Xs_filtered = scaler.transform(df_full)[:, top_indices]
        
        probs = model.predict_proba(Xs_filtered)[0]
        labels = le.classes_
        
        # Convert array thresholds to dict for matching labels
        thresh_dict = dict(zip(labels, thresholds))
        best_idx = apply_margin_logic(probs, labels, thresh_dict)
        
        return probs, labels, best_idx

def apply_margin_logic(probs, labels, thresh_dict):
    candidates = []
    for i, (prob, label) in enumerate(zip(probs, labels)):
        # Force conversion to float here
        try:
            thr = float(thresh_dict.get(label, 0.5))
        except (ValueError, TypeError):
            thr = 0.5 # Default fallback if conversion fails
            
        margin = float(prob) - thr
        if margin >= 0:
            candidates.append((i, margin))
    
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    return np.argmax(probs)
    
# --- 5. MODERN UI ---
st.set_page_config(
    page_title="Pathos AI",
    page_icon="✨",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top,
            #151515 0%,
            #0d0d0d 40%,
            #000000 100%
        );
}

.stApp > header {
    display:none;
}

.main-title{
    text-align:center;
    font-size:2.8rem;
    letter-spacing:2px;
    font-weight:800;
    color:white;
    margin-bottom:0.1rem;

    text-shadow:
    0 0 10px rgba(255,255,255,0.08);
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:0.4rem;
}

[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:24px !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    background:rgba(20,20,20,0.75) !important;
    backdrop-filter: blur(18px);
    padding:12px;
}

.stButton > button{
    width:100%;
    height:55px;
    border-radius:15px;
    font-size:18px;
    font-weight:700;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="metric-container"]{
    border-radius:15px;
    padding:15px;
    background:rgba(255,255,255,0.04);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.sidebar-card-blue{
    background:#0f172a;
    border-left:4px solid #3b82f6;
    padding:12px;
    border-radius:8px;
}

.sidebar-card-green{
    background:#0f172a;
    border-left:4px solid #22c55e;
    padding:12px;
    border-radius:8px;
}

.sidebar-card-orange{
    background:#0f172a;
    border-left:4px solid #f59e0b;
    padding:12px;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- RESULT DIALOG ----------------
@st.dialog("📊 Emotion Analysis Results", width="large")
def show_results_dialog(probs, labels, best_idx):
    # We now use best_idx instead of calculating np.argmax(probs)
    # This ensures the UI respects the margin-based thresholding logic
    detected_emotion = labels[best_idx]
    confidence = probs[best_idx]

    emotion_emojis = {
        "joy": "😊", "happy": "😊",
        "sadness": "😢", "sad": "😢",
        "anger": "😠", "fear": "😨",
        "neutral": "😐", "surprise": "😲"
    }

    emoji = emotion_emojis.get(detected_emotion.lower(), "🎭")

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:2px;
            border-radius:15px;
            background:rgba(255,255,255,0.04);
            margin-bottom:20px;
        ">
            <h1>{emoji}</h1>
            <h1 style="
                font-size:42px;
                font-weight:800;
                margin-bottom:5px;
                ">
                {detected_emotion.upper()}
                </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(confidence))
    confidence = probs[best_idx]

    st.success(
        f"Confidence Score: {confidence:.2%}"
    )

    df_results = pd.DataFrame({
        "Emotion": labels,
        "Confidence": probs
    })


    fig = px.bar(
        df_results,
        x="Confidence",
        y="Emotion",
        orientation="h",
        color="Confidence",
        color_continuous_scale="Bluered"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)



# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="main-title">
        ✨ PATHOS AI
    </div>

    <div class="subtitle">
        Understanding Human Emotions Through Speech & Text
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")
# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("## Pathos AI")

    with st.container(border=True):

        st.markdown(
            "<h4 style='text-align:center;'>Select Model</h4>",
            unsafe_allow_html=True
        )

        target_lang = st.radio(
            "",
            ["Malayalam", "Punjabi"],
            horizontal=True,
            label_visibility="collapsed"
        )

    with st.expander("🔵 About"):

        st.markdown(
            """
            <div class="sidebar-card-blue">
            Pathos AI is a multimodal emotion recognition
            platform designed for Indic languages.
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("🟢 Selected Model", expanded=True):

        if target_lang == "Malayalam":

            st.markdown("""
            <div class="sidebar-card-green">
            <h4 style="margin-top:0;">🇮🇳 Malayalam</h4>

            • Audio + Text Fusion  
            • Multilingual BERT  
            • Ensemble Learning  
            • Real-Time Inference
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(
                """
                <div class="sidebar-card-green">
                <b>Punjabi Model</b><br><br>
                • Audio Features<br>
                • Prosodic Features<br>
                • Voice Quality Metrics
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.expander("🟠 Supported Inputs"):

        st.markdown(
        """
        <div class="sidebar-card-orange">
        <b>Malayalam</b><br>
        • WAV Upload<br>
        • Voice Recording<br>
        • Text Input<br><br>

        <b>Punjabi</b><br>
        • MP3 Upload<br>
        • Voice Recording
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
    """
    <div style="
        text-align:center;
        color:#8a8a8a;
        font-size:14px;
        margin-top:10px;
    ">
        Pathos AI v1.0
    </div>
    """,
    unsafe_allow_html=True
    )

# ---------------- INPUT CARD ----------------
with st.container(border=True):
    
    col1, spacer, col2 = st.columns([1,0.08,1])

    with col1:

        st.markdown("### 🎙 Audio Input")

        input_source = st.radio(
            "",
            ["🎤 Record Audio", "📂 Upload Audio"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_source == "🎤 Record Audio":

            audio_input = mic_recorder(
                key=f"recorder_{target_lang}",
                start_prompt="🎤 Record",
                stop_prompt="⏹️ Stop"
            )

            uploaded_file = None

            if audio_input:
                st.audio(audio_input["bytes"])

        else:

            audio_input = None

            if target_lang == "Punjabi":

                uploaded_file = st.file_uploader(
                    "Upload Punjabi Audio (MP3)",
                    type=["mp3"]
                )

            else:

                uploaded_file = st.file_uploader(
                    "Upload Malayalam Audio (WAV)",
                    type=["wav"]
                )

            if uploaded_file:
                st.audio(uploaded_file)

    with col2:
        st.markdown("### 📝 Text Input")
        if target_lang == "Malayalam":
            text_input = st.text_area("Enter Malayalam Text", height=250)
        else:
            text_input = None
            st.info("Punjabi model uses audio only.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Button Column
    left, center, right = st.columns([1.5,2,1.5])
    with center:
        analyze = st.button(
                    "🚀 Analyze Emotion",
                    type="primary",
                    use_container_width=True
        )

# --- Logic after the container ---

if analyze:

    # Nothing provided
    if not uploaded_file and not audio_input:
        st.warning(
            "Please upload or record audio first."
        )
        st.stop()

    try:

        # -------------------------
        # CASE 1: MICROPHONE INPUT
        # -------------------------
        if audio_input:

            if len(audio_input["bytes"]) == 0:
                st.error("Recording is empty.")
                st.stop()

            with open("recorded_audio.webm", "wb") as f:
                f.write(audio_input["bytes"])
            
            

            audio = AudioSegment.from_file(
                "recorded_audio.webm",
                format="webm"
            )

            if target_lang == "Punjabi":

                audio.export(
                    "recorded_audio.mp3",
                    format="mp3"
                )

                temp_file = "recorded_audio.mp3"

            else:

                audio.export(
                    "recorded_audio.wav",
                    format="wav"
                )

                temp_file = "recorded_audio.wav"

        # -------------------------
        # CASE 2: FILE UPLOAD
        # -------------------------
        else:

            ext = uploaded_file.name.split(".")[-1].lower()

            temp_file = f"uploaded_audio.{ext}"

            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())
            

        # Verify audio loads correctly
        y, sr = librosa.load(
            temp_file,
            sr=None,
            mono=True
        )

    except Exception as e:

        import traceback

        st.error(
            f"Audio processing failed: {str(e)}"
        )

        st.code(traceback.format_exc())
        st.stop()

    with st.spinner(
        "Analyzing emotional content..."
    ):

        probs, labels, best_idx = run_prediction(
            temp_file,
            text_input
        )

    if probs is None or labels is None:
        st.stop()

    show_results_dialog(
        probs,
        labels,
        best_idx
    )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:13px;
        padding:10px;
    ">
        Pathos AI © 2026 • Multimodal Emotion Recognition for Indic Languages
    </div>
    """,
    unsafe_allow_html=True
)