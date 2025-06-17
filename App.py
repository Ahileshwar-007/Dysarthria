import streamlit as st
import os
import tempfile
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
import google.generativeai as genai
from utils_main import analyze_pronunciation
import queue
import time
import threading
import io
import base64
import streamlit.components.v1 as components

# Initialize session states
if "page" not in st.session_state:
    st.session_state.page = "home"
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recordings" not in st.session_state:
    st.session_state.recordings = []
if "issues" not in st.session_state:
    st.session_state.issues = []
if "generated_phoneme_pages" not in st.session_state:
    st.session_state.generated_phoneme_pages = {}
if "selected_phoneme" not in st.session_state:
    st.session_state.selected_phoneme = None
if "practice_recordings" not in st.session_state:
    st.session_state.practice_recordings = []
if "last_gpt_response" not in st.session_state:
    st.session_state.last_gpt_response = ""
if "ref_audio_path" not in st.session_state:
    st.session_state.ref_audio_path = ""
if "ref_features" not in st.session_state:
    st.session_state.ref_features = None
if "status_text" not in st.session_state:
    st.session_state.status_text = ""
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "recording_thread" not in st.session_state:
    st.session_state.recording_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = []
if "generated_prompt_history" not in st.session_state:
    st.session_state.generated_prompt_history = {}

genai.configure(api_key=st.secrets["gemini"]["api_key"])
GEMINI_MODEL = "models/gemini-1.5-flash"

st.set_page_config(page_title="Dysarthria Therapy", layout="centered")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec = librosa.feature.spectral_contrast(y=y, sr=sr)

    min_len = min(mfcc.shape[1], chroma.shape[1], spec.shape[1])
    mfcc = mfcc[:, :min_len]
    chroma = chroma[:, :min_len]
    spec = spec[:, :min_len]

    features = np.vstack([mfcc, chroma, spec])
    return features.T

def generate_prompt(phoneme, level):
    """Generate prompt with more sophisticated avoidance"""
    level = max(1, min(5, level))
    
    if phoneme not in st.session_state.generated_prompt_history:
        st.session_state.generated_prompt_history[phoneme] = []
    
    history = st.session_state.generated_prompt_history[phoneme]
    max_history_size = 10
    
    level_prompts = {
        1: f"Generate a single word containing the /{phoneme}/ sound.",
        2: f"Give me 3-4 words using /{phoneme}/ (don't repeat exact phrases).",
        3: f"Create a unique sentence with /{phoneme}/ sound.",
        4: f"Generate an advanced sentence with /{phoneme}/ (avoid common examples).",
        5: f"Invent a creative tongue-twister featuring /{phoneme}/ sound."
    }
    
    exclusion_rules = {
        1: f" STRICTLY avoid these exact words: {', '.join(history) if history else 'none yet'}",
        2: f" Avoid these exact word combinations: {', '.join(history) if history else 'none yet'}",
        3: f" Don't use these exact sentences: {', '.join(history) if history else 'none yet'}",
        4: f" These advanced sentences were already used: {', '.join(history) if history else 'none yet'}",
        5: f" These tongue-twisters were already created: {', '.join(history) if history else 'none yet'}"
    }
    
    full_prompt = level_prompts[level] + exclusion_rules[level] + " Be creative and provide something completely different."
    
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(full_prompt)
        new_prompt = response.text.strip()
        
        # For level 1, check if any word was repeated despite our instructions
        if level == 1 and history:
            new_word = new_prompt.lower().strip('.,!?')
            if any(h.lower().strip('.,!?') == new_word for h in history):
                # If Gemini failed to follow instructions, rotate history
                oldest = history.pop(0)
                history.append(oldest)
                return oldest, level
        
        history.append(new_prompt)
        if len(history) > max_history_size:
            history.pop(0)
        
        return new_prompt, level
    
    except Exception as e:
        if history:
            return history[-1], level
        return f"Error: {str(e)}", level
    
def speak_once(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        audio_html = f"""
        <audio id="player" controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            function playAudio() {{
                const audio = document.getElementById('player');
                audio.play().catch(e => console.log('Autoplay prevented:', e));
            }}
            document.addEventListener('click', playAudio, {{once: true}});
            setTimeout(playAudio, 500);
        </script>
        """
        components.html(audio_html, height=60)
    except Exception as e:
        st.error(f"Speech error: {e}")

def record_audio(stop_event):
    samplerate = 44100
    channels = 1
    
    def callback(indata, frames, time, status):
        if stop_event.is_set():
            raise sd.CallbackAbort
        st.session_state.audio_queue.put(indata.copy())
    
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

def start_recording():
    st.session_state.recording = True
    st.session_state.status_text = "Recording..."
    st.session_state.audio_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    st.session_state.recording_thread = threading.Thread(
        target=record_audio,
        args=(st.session_state.stop_event,)
    )
    st.session_state.recording_thread.start()

def process_practice_recording(file_path):
    """Process a recording (either from microphone or file upload)"""
    try:
        # Use the latest generated prompt as transcript (remove prefix if present)
        target_text = st.session_state.last_gpt_response
        if target_text.startswith("Record this: "):
            target_text = target_text[len("Record this: "):]
        elif target_text.startswith("Repeat after me: "):
            target_text = target_text[len("Repeat after me: "):]
            
        issues = analyze_pronunciation(file_path, target_text)
        
        is_good_recording = not (isinstance(issues, list) and issues and 
                               not (len(issues) == 1 and issues[0] == "Recording could not be analyzed"))
        
        st.session_state.practice_recordings.append({
            "path": file_path,
            "target_text": target_text,
            "user_text": target_text,  # Using target as transcript
            "issues": issues,
            "is_good": is_good_recording
        })
        
        if st.session_state.selected_phoneme:
            update_phoneme_level(st.session_state.selected_phoneme, is_good_recording)
    except Exception as e:
        target_text = st.session_state.last_gpt_response
        if target_text.startswith("Record this: "):
            target_text = target_text[len("Record this: "):]
        elif target_text.startswith("Repeat after me: "):
            target_text = target_text[len("Repeat after me: "):]
            
        st.session_state.practice_recordings.append({
            "path": file_path,
            "target_text": target_text,
            "user_text": target_text,  # Using target as transcript
            "issues": ["Recording could not be analyzed"],
            "is_good": False
        })
        if st.session_state.selected_phoneme:
            update_phoneme_level(st.session_state.selected_phoneme, False)

def stop_recording():
    if st.session_state.recording:
        st.session_state.recording = False
        st.session_state.stop_event.set()
        if st.session_state.recording_thread:
            st.session_state.recording_thread.join()
        
        st.session_state.status_text = "Processing..."
        audio_data = []
        while not st.session_state.audio_queue.empty():
            audio_data.append(st.session_state.audio_queue.get())
        
        if audio_data:
            audio_np = np.concatenate(audio_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                sf.write(temp_audio_file.name, audio_np, 44100)
                # Use fixed transcript for home page recordings
                text = "The quick brown fox jumps over the lazy dog"
                issues = analyze_pronunciation(temp_audio_file.name, text)
                st.session_state.recordings.append(temp_audio_file.name)
                st.session_state.issues.append(issues)
        
        st.session_state.status_text = ""
        st.rerun()

def update_phoneme_level(phoneme, is_good_recording):
    """Update level based on consecutive good recordings."""
    if phoneme not in st.session_state.generated_phoneme_pages:
        return
    
    phoneme_data = st.session_state.generated_phoneme_pages[phoneme]
    phoneme_data.setdefault("consecutive_good", 0)
    phoneme_data.setdefault("level", 1)
    
    if is_good_recording:
        phoneme_data["consecutive_good"] += 1
        if phoneme_data["consecutive_good"] >= 5:
            phoneme_data["level"] = min(5, phoneme_data["level"] + 1)
            phoneme_data["consecutive_good"] = 0
            st.toast(f"Level upgraded to {phoneme_data['level']} for /{phoneme}/!")
    else:
        phoneme_data["consecutive_good"] = 0
    
    st.rerun()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dysarthria_model.keras")

model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### 👤 User")
    st.markdown("**Dysarthria Severity:** Moderate")
    
    if st.button("🏠 Home"):
        st.session_state.page = "home"
    
    if st.session_state.generated_phoneme_pages:
        st.markdown("### 🎯 Practice Targets")
        for phoneme in st.session_state.generated_phoneme_pages.keys():
            phoneme_data = st.session_state.generated_phoneme_pages[phoneme]
            current_level = phoneme_data.get("level", 1)
            consecutive_good = phoneme_data.get("consecutive_good", 0)
            
            col1, col2, col3 = st.columns([3, 1, 1]) 
            with col1:
                if st.button(f"/{phoneme}/", key=f"sidebar_practice_{phoneme}"):
                    st.session_state.selected_phoneme = phoneme
                    st.session_state.page = "practice"
            with col2:
                st.markdown(f"**Lvl {current_level}**")
            with col3:
                st.progress(min(1.0, consecutive_good / 5))

# Home Page
if st.session_state.page == "home":
    st.markdown("""
        <style>
            .dropdown-button {
                background-color: #f1f1f1;
                border: none;
                padding: 8px 16px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 10px;
                color: black;
            }
            .dropdown-content {
                display: none;
                position: absolute;
                background-color: #f9f9f9;
                min-width: 200px;
                box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
                padding: 12px 16px;
                border-radius: 10px;
                z-index: 9999;
                color: black;
            }
            .dropdown-wrapper:hover .dropdown-content {
                display: block;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 Dysarthria Sound Cue Analyzer")
    status_box = st.empty()
    status_box.info("🟡 Waiting to start...")

    uploaded_file = st.file_uploader("📤 Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        status_box.info("⏳ Processing uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
    
        st.session_state.recordings.append(temp_file_path)
        try:
            # Use fixed transcript instead of speech recognition
            text = "The quick brown fox jumps over the lazy dog"
            issues = analyze_pronunciation(temp_file_path, text)
            st.session_state.issues.append(issues)
        except Exception as e:
            st.session_state.issues.append([f"Error: {str(e)}"])
        status_box.success("✅ Uploaded file processed.")

    if st.button("⏺️ Start Recording" if not st.session_state.recording else "⏹️ Stop Recording"):
        if st.session_state.recording:
            stop_recording()
        else:
            start_recording()

    if st.session_state.recording:
        time.sleep(0.1)
        st.rerun()

    if st.session_state.recordings:
        st.subheader("🎧 Recordings")
        for i, rec_path in enumerate(st.session_state.recordings):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.audio(rec_path)
            with col2:
                issues = st.session_state.issues[i] if i < len(st.session_state.issues) else []
                dropdown_html = f"""
                <div class="dropdown-wrapper">
                    <button class="dropdown-button">🔍 Issues</button>
                    <div class="dropdown-content">
                        {'<ul>' + ''.join(f'<li>{cue}</li>' for cue in issues) + '</ul>' if issues else '<p>✅ No issues</p>'}
                    </div>
                </div>
                """
                st.markdown(dropdown_html, unsafe_allow_html=True)
            with col3:
                if st.button(f"Practice", key=f"practice_recording_{i}"):
                    st.session_state.generated_phoneme_pages = {}
                    for phoneme in st.session_state.issues[i]:
                        st.session_state.generated_phoneme_pages[phoneme] = {
                            "mean": 0.0, 
                            "count": 0, 
                            "history": [],
                            "level": 1,
                            "consecutive_good": 0
                        }
                    st.success("✅ Practice pages created!")
                    st.rerun()
            with col4:
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    try:
                        os.remove(rec_path)
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                    st.session_state.recordings.pop(i)
                    if i < len(st.session_state.issues):
                        st.session_state.issues.pop(i)
                    st.rerun()
    else:
        st.info("🎙️ No recordings yet.")

# Practice Page
elif st.session_state.page == "practice" and st.session_state.selected_phoneme:
    phoneme = st.session_state.selected_phoneme
    phoneme_data = st.session_state.generated_phoneme_pages[phoneme]
    current_level = phoneme_data.get("level", 1)
    consecutive_good = phoneme_data.get("consecutive_good", 0)
    
    st.markdown(f"# Practice Session: /{phoneme}/")
    
    col_level, col_progress = st.columns([1, 4])
    with col_level:
        st.markdown(f"### Level: {current_level}/5")
    with col_progress:
        st.markdown("**Progress to next level:**")
        st.progress(min(1.0, consecutive_good / 5))
        st.caption(f"{consecutive_good}/5 good recordings needed for next level")
    
    if not st.session_state.last_gpt_response:
        gpt_sentence, _ = generate_prompt(phoneme, current_level)
        st.session_state.last_gpt_response = f"Record this: {gpt_sentence}"
        ref_audio_path = os.path.join(tempfile.gettempdir(), f"reference_{phoneme}.mp3")
        tts = gTTS(text=gpt_sentence, lang='en')
        tts.save(ref_audio_path)
        st.session_state.ref_audio_path = ref_audio_path
        st.session_state.ref_features = extract_features(ref_audio_path)

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("⏺️ Start Recording", disabled=st.session_state.recording):
            start_recording()
    with colB:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
            stop_recording()
    with colC:
        if st.session_state.status_text:
            st.markdown(f"**{st.session_state.status_text}**")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✨ Generate"):
            gpt_sentence, _ = generate_prompt(phoneme, current_level)
            st.session_state.last_gpt_response = f"Record this: {gpt_sentence}"
            ref_audio_path = os.path.join(tempfile.gettempdir(), f"reference_{phoneme}.mp3")
            tts = gTTS(text=gpt_sentence, lang='en')
            tts.save(ref_audio_path)
            st.session_state.ref_audio_path = ref_audio_path
            st.session_state.ref_features = extract_features(ref_audio_path)
    with col2:
        if st.button("🔊 Speaker"):
            speak_once(st.session_state.last_gpt_response)

    st.markdown(f"""
        <div style='padding: 1rem; background-color: #000000; border-radius: 0.5rem;
                    font-size: 1.2rem; border-left: 5px solid #4a90e2; color: white;'>
            {st.session_state.last_gpt_response}
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload a .wav file", type=["wav"], key="file_uploader_prac")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        process_practice_recording(temp_file_path)
        st.rerun()

    if st.session_state.practice_recordings:
        st.subheader("🎧 Practice Recordings")
        for rec in reversed(st.session_state.practice_recordings):
            cols = st.columns([5, 2])
            with cols[0]:
                st.audio(rec["path"])
            with cols[1]:
                issues = rec.get("issues", [])
                has_real_issues = (isinstance(issues, list) and issues and 
                                  not (len(issues) == 1 and issues[0] == "Recording could not be analyzed"))
                status = "🔴 **Dysarthric**" if has_real_issues else "🟢 **Non Dysarthric**"
                st.markdown(status)
    else:
        st.info("🎙️ No practice recordings yet.")