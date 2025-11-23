import os
import re
import nltk
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import soundfile as sf
import sys

# Setup NLTK
nltk_data_path = os.path.expanduser("nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)
stop_words = set(stopwords.words("english"))

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Record audio
def record_audio(filename="user_input.wav", duration=5, fs=16000):
    print("🔴 Recording for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    return filename

# Transcribe audio to text
def transcribe_audio(filepath):
    speech, rate = sf.read(filepath)
    if len(speech.shape) > 1:
        speech = speech[:, 0]
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()
    return transcription

# Search corpus using regex at sentence level
def search_corpus(transcription, corpus_text, top_n=3):
    tokens = word_tokenize(transcription)
    keywords = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    sentences = sent_tokenize(corpus_text)
    matches = []

    for sentence in sentences:
        score = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", sentence.lower()))
        if score > 0:
            matches.append((sentence.strip(), score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return [sent for sent, _ in matches[:top_n]]

# Main function
def main():
    print("📌 Paste your full corpus below. Press Ctrl+Z to submit:")
    corpus_text = sys.stdin.read()
    print(f"✅ Corpus loaded. Total characters: {len(corpus_text)}\n")

    audio_path = record_audio()
    print("🎤 Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    print(f"\n📝 Transcription: {transcription.upper()}")

    print("\n🔍 Top Matches:")
    results = search_corpus(transcription, corpus_text)
    if results:
        for i, line in enumerate(results, 1):
            print(f"{i}. {line}")
    else:
        print("No matching sentences found.")

if __name__ == "__main__":
    main()
