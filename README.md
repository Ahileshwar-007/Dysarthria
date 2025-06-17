# 🗣️ Dysarthria Detection and Speech Therapy Assistant

This project is an AI-powered web application built using **Streamlit** that detects dysarthria in speech and provides **personalized speech therapy**. It analyzes pronunciation from user-recorded audio and guides therapy using interactive prompts based on phoneme clarity.

---

## 🚀 Features

- 🎙️ **Record or Upload Speech**: Accepts spoken input (5–10 seconds) for analysis.
- 🤖 **Dysarthria Detection**: Uses a trained neural network to identify speech impairments.
- 🔍 **Phoneme-Level Feedback**: Highlights mispronounced phonemes and provides focused practice.
- 📚 **Therapy Prompts**: Includes minimal pairs, rhyming, contrastive stress, and Q&A exercises.
- 🌐 **Streamlit Frontend**: Clean and interactive UI with dropdown-based phoneme navigation.
- 🔄 **Dynamic Feedback**: Prompts change based on clarity level (tracked over time).

---

## 🧠 Tech Stack

- **Python**, **TensorFlow/Keras**
- **Streamlit** for frontend
- **Librosa**, **SoundDevice**, **SoundFile** for audio processing
- **Google Generative AI API** for therapy content generation
- **NLTK** & **CMUdict** for phoneme-level linguistic mapping