import librosa
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import cmudict
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import os

nltk.download('cmudict')
pron_dict = cmudict.dict()

model = tf.keras.models.load_model("dysarthria_model.keras")
SAMPLE_RATE = 22050

def extract_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        n_fft = min(2048, len(audio) - 1)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft)
        mfcc_mean = np.mean(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft)
        chroma_mean = np.mean(chroma, axis=1)

        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)

        feature_vector = np.concatenate([mfcc_mean, chroma_mean, spec_contrast_mean])
        return feature_vector

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

def get_phonemes(word):
    word = word.lower()
    if word in pron_dict:
        phonemes = pron_dict[word][0] 
        return [p.strip("012") for p in phonemes]  
    return []


def analyze_pronunciation(audio_path, transcript):
    problematic_phonemes = set()

    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-40)

    words = transcript.split()
    if len(chunks) != len(words):
        print("Warning: mismatch between audio chunks and transcript words")

    min_len = min(len(chunks), len(words))

    for i in range(min_len):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            chunk_path = f.name
            chunks[i].export(chunk_path, format="wav")

        features = extract_features(chunk_path)
        os.remove(chunk_path)

        if features is None:
            continue

        features = np.expand_dims(features, axis=0)  
        prediction = model.predict(features)[0][0]  

        if prediction > 0.5:
            phonemes = get_phonemes(words[i])
            problematic_phonemes.update(phonemes)

    return sorted(list(problematic_phonemes))
