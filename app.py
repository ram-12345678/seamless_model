import streamlit as st
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np
import torch

# Initialize processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Language dictionary for selection
lang_dict = {
    'English': 'eng',
    'Russian': 'hi',
    'Spanish': 'spa',
    # Add more languages as needed
}

def translate_text(source_lang, target_lang, text_input):
    text_inputs = processor(text=text_input, src_lang=lang_dict[source_lang], return_tensors="pt")
    translated_audio = model.generate(**text_inputs, tgt_lang=lang_dict[target_lang])[0].cpu().numpy().squeeze()
    return (16000, translated_audio)

def translate_audio(source_lang, target_lang, audio):
    audio = np.array(audio)
    audio = np.expand_dims(audio, axis=0)  # Add batch dimension
    audio = torch.from_numpy(audio)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    translated_audio = model.generate(**audio_inputs, tgt_lang=lang_dict[target_lang])[0].cpu().numpy().squeeze()
    return (16000, translated_audio)

# Streamlit interface
st.title("Speech-to-Speech Translation")

source_lang = st.selectbox("Select Source Language", list(lang_dict.keys()))
target_lang = st.selectbox("Select Target Language", list(lang_dict.keys()))

tab1, tab2 = st.tabs(["Text", "Audio"])

with tab1:
    text_input = st.text_area("Enter Text to Translate")
    if st.button("Translate Text"):
        sample_rate, translated_audio = translate_text(source_lang, target_lang, text_input)
        st.audio(translated_audio, sample_rate=sample_rate)

with tab2:
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio, sample_rate = torchaudio.load(uploaded_file)
        if st.button("Translate Audio"):
            sample_rate, translated_audio = translate_audio(source_lang, target_lang, audio)
            st.audio(translated_audio, sample_rate=sample_rate)
